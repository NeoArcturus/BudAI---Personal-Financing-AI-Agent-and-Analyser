import json
import uuid
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy import text
from config import SessionLocal, redis_client
from models.status_codes import MacroStatus
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

ONS_BASE_URL = "https://api.beta.ons.gov.uk/v1"
ONS_DATASET_ID = "cpih01"
ONS_EDITION = "time-series"
GEOGRAPHY_UK = "K02000001"

# Target COICOP indicators to monitor
TARGET_AGGREGATES: Dict[str, str] = {
    "CP00": "Headline CPIH Overall Index",
    "CP01": "Food and non-alcoholic beverages",
    "CP011": "Food",
    "CP0111": "Bread and cereals",
    "CP0114": "Milk, cheese and eggs",
    "CP03": "Clothing and footwear",
    "CP04": "Housing, water, electricity, gas and other fuels",
    "CP041": "Actual rentals for housing",
    "CP042": "Owner Occupiers Housing Costs",
    "CP045": "Electricity, gas and other fuels",
    "CP0451": "Electricity",
    "CP0452": "Gas",
    "CP049": "Council Tax and rates",
    "CP06": "Health and medical",
    "CP07": "Transport",
    "CP0722": "Vehicle fuels and lubricants",
    "CP0731": "Passenger transport by railway",
    "CP08": "Communication and telecom",
    "CP09": "Recreation and culture",
    "CP11": "Restaurants and hotels",
    "CP111": "Catering services and dining out"
}

# Mapping from BudAI friendly tags to ONS codes
TAG_TO_ONS_CODE: Dict[str, str] = {
    "#ons:cp00": "CP00",
    "#ons:cp01": "CP01",
    "#ons:cp01_food": "CP011",
    "#ons:cp03_apparel": "CP03",
    "#ons:cp04": "CP04",
    "#ons:cp04_housing": "CP041",
    "#ons:cp04_energy": "CP045",
    "#ons:cp04_council": "CP049",
    "#ons:cp06_health": "CP06",
    "#ons:cp07": "CP07",
    "#ons:cp07_fuel": "CP0722",
    "#ons:cp07_transit": "CP0731",
    "#ons:cp08_telecom": "CP08",
    "#ons:cp09": "CP09",
    "#ons:cp09_digital": "CP09",
    "#ons:cp09_leisure": "CP09",
    "#ons:cp11": "CP11"
}

CATEGORY_TO_ONS_CODE: Dict[str, str] = {
    "Food & Dining": "CP01",
    "Utilities": "CP045",
    "Transportation": "CP07",
    "Housing": "CP041",
    "Subscriptions & Digital Services": "CP08",
    "Entertainment & Lifestyle": "CP09",
    "Shopping & Retail": "CP03",
    "Healthcare": "CP06"
}

class ONSClient:
    """
    High-Performance ONS Macroeconomic Client.
    - Zero API keys required (consumes public UK Office for National Statistics Beta API).
    - Multi-tiered caching: L1 Process Memory -> L2 Upstash Redis -> L3 Neon PostgreSQL.
    - Calendar-aligned monthly batch synchronization with HTTP conditional headers (ETags).
    """

    def __init__(self):
        self.redis = redis_client
        self.http_client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "BudAI-Autonomous-Finance-Agent/1.0"}
        )
        self._l1_cache: Dict[str, Any] = {}
        self._l1_cache_time: Optional[datetime] = None

    def _get_latest_version_metadata(self) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Fetches the latest version number, release date, and ETag for cpih01."""
        url = f"{ONS_BASE_URL}/datasets/{ONS_DATASET_ID}/editions/{ONS_EDITION}/versions"
        try:
            resp = self.http_client.get(url)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if not items:
                return None, None, None
            latest = items[0]
            version = latest.get("version")
            release_date = latest.get("release_date")
            etag = resp.headers.get("ETag") or f"ons-v{version}-{release_date}"
            return version, release_date, etag
        except Exception as e:
            logger.error(f"Failed to fetch ONS version list: {e}")
            return None, None, None

    def _get_time_options(self, version: int) -> List[str]:
        """Fetches available time periods (e.g. ['Jan-26', 'Dec-25', ...])."""
        url = f"{ONS_BASE_URL}/datasets/{ONS_DATASET_ID}/editions/{ONS_EDITION}/versions/{version}/dimensions/time/options?limit=30"
        try:
            resp = self.http_client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return [it.get("option") for it in data.get("items", []) if it.get("option")]
        except Exception as e:
            logger.error(f"Failed to fetch ONS time options: {e}")
            return []

    def _fetch_observation(self, version: int, aggregate: str, time_period: str) -> Optional[float]:
        """Queries single observation value for a given aggregate and time period."""
        url = (
            f"{ONS_BASE_URL}/datasets/{ONS_DATASET_ID}/editions/{ONS_EDITION}/"
            f"versions/{version}/observations?aggregate={aggregate}&geography={GEOGRAPHY_UK}&time={time_period}"
        )
        try:
            resp = self.http_client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                obs = data.get("observations", [])
                if obs and "observation" in obs[0]:
                    return float(obs[0]["observation"])
        except Exception as e:
            logger.debug(f"Observation fetch miss for {aggregate} at {time_period}: {e}")
        return None

    def sync_telemetry(self, force: bool = False) -> Dict[str, Any]:
        """
        Executes monthly synchronization:
        1. Checks if a new release version exists.
        2. Queries target aggregates for current period, previous month, and previous year.
        3. Computes YoY and MoM percentage changes.
        4. Writes to Neon DB and Upstash Redis.
        """
        logger.info("Executing ONS Macroeconomic Telemetry Sync...")

        version, release_date, etag = self._get_latest_version_metadata()
        if not version:
            return {
                "status": "error",
                "status_code": MacroStatus.TELEMETRY_FETCH_ERROR.value,
                "message": "Unable to contact ONS Beta API"
            }

        # Check if already synced in Redis
        if not force and self.redis:
            cached_version = self.redis.get("macro:ons:version")
            if cached_version and int(cached_version) == version:
                logger.info(f"ONS telemetry is already up-to-date (Version {version}, {release_date}).")
                return {
                    "status": "not_modified",
                    "status_code": MacroStatus.TELEMETRY_NOT_MODIFIED.value,
                    "version": version,
                    "release_date": release_date
                }

        time_options = self._get_time_options(version)
        if len(time_options) < 13:
            logger.warning("Insufficient time periods returned by ONS; aborting sync.")
            return {
                "status": "error",
                "status_code": MacroStatus.TELEMETRY_FETCH_ERROR.value,
                "message": "Insufficient time series depth"
            }

        current_period = time_options[0]       # e.g. "Jan-26"
        prev_month_period = time_options[1]    # e.g. "Dec-25"
        prev_year_period = time_options[12]    # e.g. "Jan-25" (12 months prior)

        records_to_insert = []
        redis_hash_payload: Dict[str, str] = {
            "period": current_period,
            "release_date": release_date or "",
            "version": str(version),
            "updated_at": datetime.utcnow().isoformat()
        }

        for code, name in TARGET_AGGREGATES.items():
            val_now = self._fetch_observation(version, code, current_period)
            if val_now is None:
                continue

            val_prev_month = self._fetch_observation(version, code, prev_month_period)
            val_prev_year = self._fetch_observation(version, code, prev_year_period)

            yoy = round(((val_now - val_prev_year) / val_prev_year) * 100, 3) if val_prev_year else None
            mom = round(((val_now - val_prev_month) / val_prev_month) * 100, 3) if val_prev_month else None

            records_to_insert.append({
                "id": str(uuid.uuid4()),
                "source": "ONS",
                "dataset": ONS_DATASET_ID,
                "indicator_code": code,
                "indicator_name": name,
                "period": current_period,
                "value": val_now,
                "delta_pct_yoy": yoy,
                "delta_pct_mom": mom,
                "etag": etag,
                "metadata_json": json.dumps({"prev_month": prev_month_period, "prev_year": prev_year_period})
            })

            # Store in Redis hot cache
            redis_hash_payload[f"{code}:value"] = str(val_now)
            if yoy is not None:
                redis_hash_payload[f"{code}:yoy"] = str(yoy)
            if mom is not None:
                redis_hash_payload[f"{code}:mom"] = str(mom)

        # 1. Persist to Neon PostgreSQL
        with SessionLocal() as session:
            insert_sql = text("""
                INSERT INTO macro_telemetry (
                    id, source, dataset, indicator_code, indicator_name, period,
                    value, delta_pct_yoy, delta_pct_mom, etag, metadata_json, created_at
                ) VALUES (
                    :id, :source, :dataset, :indicator_code, :indicator_name, :period,
                    :value, :delta_pct_yoy, :delta_pct_mom, :etag, CAST(:metadata_json AS JSONB), NOW()
                ) ON CONFLICT (source, dataset, indicator_code, period) 
                DO UPDATE SET
                    value = EXCLUDED.value,
                    delta_pct_yoy = EXCLUDED.delta_pct_yoy,
                    delta_pct_mom = EXCLUDED.delta_pct_mom,
                    etag = EXCLUDED.etag;
            """)
            for r in records_to_insert:
                session.execute(insert_sql, r)
            session.commit()

        # 2. Update Upstash Redis
        if self.redis:
            try:
                self.redis.hset("macro:ons:latest", mapping=redis_hash_payload)
                self.redis.expire("macro:ons:latest", 3024000)  # 35 days sliding TTL
                self.redis.set("macro:ons:version", str(version))
                if etag:
                    self.redis.set("macro:ons:etag", etag)
            except Exception as e:
                logger.error(f"Failed to update Redis macro cache: {e}")

        # 3. Invalidate L1 process memory
        self._l1_cache = redis_hash_payload
        self._l1_cache_time = datetime.utcnow()

        logger.info(
            f"ONS Macro Telemetry sync completed: {len(records_to_insert)} indicators saved "
            f"for period {current_period} (Version {version})."
        )

        return {
            "status": "success",
            "status_code": MacroStatus.TELEMETRY_SYNCED.value,
            "period": current_period,
            "version": version,
            "indicators_synced": len(records_to_insert)
        }

    def get_category_inflation(self, category_or_tag: str) -> float:
        """
        Sub-millisecond query returning the latest YoY inflation percentage for a given
        BudAI category, sub-category tag, or raw ONS code.
        
        Example:
            get_category_inflation("Food & Dining") -> 3.52
            get_category_inflation("#ons:cp04_energy") -> 8.14
        """
        # Resolve code
        code = None
        if category_or_tag in TARGET_AGGREGATES:
            code = category_or_tag
        elif category_or_tag in TAG_TO_ONS_CODE:
            code = TAG_TO_ONS_CODE[category_or_tag]
        elif category_or_tag in CATEGORY_TO_ONS_CODE:
            code = CATEGORY_TO_ONS_CODE[category_or_tag]
        else:
            code = "CP00"  # Default to Headline Inflation

        # 1. Try L1 memory cache (< 0.1ms)
        if self._l1_cache and f"{code}:yoy" in self._l1_cache:
            try:
                return float(self._l1_cache[f"{code}:yoy"])
            except ValueError:
                pass

        # 2. Try Redis (< 2ms)
        if self.redis:
            try:
                val = self.redis.hget("macro:ons:latest", f"{code}:yoy")
                if val:
                    return float(val)
            except Exception as e:
                logger.debug(f"Redis macro lookup failed: {e}")

        # 3. Fallback to Neon PostgreSQL (~20ms)
        try:
            with SessionLocal() as session:
                row = session.execute(
                    text("""
                        SELECT delta_pct_yoy 
                        FROM macro_telemetry 
                        WHERE dataset = :ds AND indicator_code = :code 
                        ORDER BY created_at DESC LIMIT 1
                    """),
                    {"ds": ONS_DATASET_ID, "code": code}
                ).fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception as e:
            logger.error(f"PostgreSQL macro lookup failed: {e}")

        # Static fallback if database has zero macro records
        return 2.8  # Standard UK baseline inflation assumption

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Returns the full parsed dictionary of latest macro telemetry."""
        if self.redis:
            try:
                data = self.redis.hgetall("macro:ons:latest")
                if data:
                    return data
            except Exception:
                pass
        return self._l1_cache or {}

_ons_client_instance: Optional[ONSClient] = None

def get_ons_client() -> ONSClient:
    global _ons_client_instance
    if _ons_client_instance is None:
        _ons_client_instance = ONSClient()
    return _ons_client_instance
