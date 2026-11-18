import json
import ctypes
import os
import subprocess
import math
import calendar
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def _ensure_compiled():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    algorithm_dir = os.path.join(base_dir, "algorithm")
    algo_cpp = os.path.join(algorithm_dir, "algorithm.cpp")
    hybrid_cpp = os.path.join(algorithm_dir, "hybrid_algorithm.cpp")
    output_so = os.path.join(base_dir, "hybrid_forecaster.so")
    
    if os.path.exists(output_so):
        try:
            ctypes.CDLL(output_so)
            return output_so
        except Exception:
            logger.warning(json.dumps({"message": f"Existing C++ engine at {output_so} is corrupt. Recompiling...", "status_code": 400}))
            try: os.remove(output_so)
            except: pass

    logger.info(json.dumps({"message": f"Compiling C++ forecaster engine to {output_so}...", "status_code": 200}))
    subprocess.run([
        "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
        f"-I{algorithm_dir}", "-o", output_so, algo_cpp, hybrid_cpp
    ], check=True)
    return output_so

def run_hybrid_engine(S0, mu, params, days, paths, account_id, deterministic_calendar):
    output_so = _ensure_compiled()
    lib = ctypes.CDLL(output_so)
    lib.run_hybrid_forecast_v3.argtypes = [
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    lib.run_hybrid_forecast_v3.restype = ctypes.c_int
    buffer_size = days + 1
    cal_ptr = (ctypes.c_double * buffer_size)(*deterministic_calendar)
    mean_out = (ctypes.c_double * buffer_size)()
    careless_out = (ctypes.c_double * buffer_size)()
    optimal_out = (ctypes.c_double * buffer_size)()
    res = lib.run_hybrid_forecast_v3(
        S0, mu,
        params.get('kappa', 2.0), params.get('theta', 0.04), params.get('xi', 0.1), params.get('rho', -0.5),
        params.get('lambda', 0.1), params.get('mu_J', -0.05), params.get('sigma_J', 0.1),
        buffer_size, paths, account_id.encode('utf-8'),
        cal_ptr, mean_out, careless_out, optimal_out
    )
    if res != 0: raise RuntimeError("Balance Engine Failed.")
    return pd.DataFrame([list(mean_out), list(careless_out), list(optimal_out)])

def run_converged_expense_engine(E0, mu, days, paths, account_id, deterministic_calendar):
    """
    Unified backward-compatible wrapper that delegates to run_hybrid_engine.
    """
    params = {'kappa': 2.0, 'theta': 0.04, 'xi': 0.1, 'rho': -0.5, 'lambda': 0.1, 'mu_J': -0.05, 'sigma_J': 0.1}
    df = run_hybrid_engine(E0, mu, params, days, paths, account_id, deterministic_calendar)
    return pd.DataFrame([df.iloc[0].tolist()])

def calculate_last_month_expense_metrics(user_uuid: str, current_dt: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculates the exact average daily expense rate for the previous calendar month
    (e.g., if today is in November, calculates for full October 1..31).
    A month is strictly a calendar month (Jan, Feb, Oct, etc.).
    If the immediately preceding calendar month had 0 recorded transactions, looks back
    to the most recent completed calendar month with spending data.
    """
    if current_dt is None:
        current_dt = datetime.now(timezone.utc)

    first_of_current_month = current_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_moment_of_previous_month = first_of_current_month - timedelta(microseconds=1)
    first_of_previous_month = last_moment_of_previous_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    days_in_last_month = calendar.monthrange(first_of_previous_month.year, first_of_previous_month.month)[1]
    last_month_name = first_of_previous_month.strftime("%B")

    total_last_month_debits = 0.0
    used_month_name = last_month_name
    used_days = days_in_last_month

    try:
        with SessionLocal() as db:
            row = db.execute(
                text("""
                    SELECT COALESCE(SUM(ABS(amount)), 0.0) as total_debits,
                           COUNT(*) as tx_count
                    FROM transactions
                    WHERE user_uuid = :user_uuid
                      AND amount < 0
                      AND date >= :start_dt
                      AND date <= :end_dt
                """),
                {
                    "user_uuid": user_uuid,
                    "start_dt": first_of_previous_month,
                    "end_dt": last_moment_of_previous_month
                }
            ).fetchone()

            if row and row[1] > 0:
                total_last_month_debits = float(row[0] or 0.0)
            else:
                recent_month_row = db.execute(
                    text("""
                        SELECT DATE_TRUNC('month', date) as m,
                               SUM(ABS(amount)) as total_debits,
                               COUNT(*) as tx_count
                        FROM transactions
                        WHERE user_uuid = :user_uuid
                          AND amount < 0
                          AND date < :current_month_start
                        GROUP BY 1
                        ORDER BY 1 DESC
                        LIMIT 1
                    """),
                    {
                        "user_uuid": user_uuid,
                        "current_month_start": first_of_current_month
                    }
                ).fetchone()
                if recent_month_row and recent_month_row[2] > 0:
                    m_dt = recent_month_row[0]
                    total_last_month_debits = float(recent_month_row[1] or 0.0)
                    used_month_name = m_dt.strftime("%B")
                    used_days = calendar.monthrange(m_dt.year, m_dt.month)[1]
    except Exception as e:
        logger.warning(f"Error querying last month debits for {user_uuid}: {e}")

    daily_rate = round(total_last_month_debits / max(1, used_days), 2)
    return {
        "last_month_name": used_month_name,
        "days_in_last_month": used_days,
        "total_debits": round(total_last_month_debits, 2),
        "daily_rate": daily_rate
    }

def generate_expense_forecast_trajectories(
    user_uuid: str,
    account_id: str,
    E0: float,
    mu_E: float,
    days: int = 30,
    surge_status: Optional[Dict[str, Any]] = None,
    deterministic_calendar: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Generates unified expense forecast trajectories:
    1. Expected / Mean spend line based on current spending rate E0 and drift mu_E (via C++ Monte Carlo)
    2. Optimal spend line based on disciplined budget rate
    3. Last month spend line based on previous calendar month total debits / days
    4. Conditional Careless spend line calculated ONLY IF a spending surge is detected (Z >= 2.0 or anomaly fired)
    """
    if deterministic_calendar is None:
        deterministic_calendar = np.zeros(days + 1)

    # 1. Expected Run Rate via C++ Jump Heston engine
    params = {'kappa': 2.0, 'theta': 0.04, 'xi': 0.1, 'rho': -0.5, 'lambda': 0.1, 'mu_J': -0.05, 'sigma_J': 0.1}
    try:
        exp_df = run_hybrid_engine(E0, mu_E, params, days, 10000, account_id, deterministic_calendar)
        raw_exp = exp_df.iloc[0].tolist() if not exp_df.empty else [E0 * math.exp(mu_E * t) for t in range(days + 1)]
    except Exception as e:
        logger.warning(f"Unified C++ expense engine fallback: {e}")
        raw_exp = [E0 * math.exp(mu_E * t) for t in range(days + 1)]

    # Future timeline slice
    trajectory_len = min(days, len(raw_exp) - 1 if len(raw_exp) > days else len(raw_exp))
    future_exp = [round(float(raw_exp[i + 1]), 2) for i in range(trajectory_len)] if len(raw_exp) > days else [round(float(v), 2) for v in raw_exp[:days]]

    # 2. Optimal Rate: Disciplined essential spending target (-25% discretionary trim)
    optimal_rate = max(1.0, round(E0 * 0.75, 2))
    mu_opt = min(0.0, mu_E - 0.01)
    optimal_trajectory = [round(float(optimal_rate * math.exp(mu_opt * t)), 2) for t in range(len(future_exp))]

    # 3. Last Month Spend Rate (Strict previous calendar month)
    last_month_metrics = calculate_last_month_expense_metrics(user_uuid)
    last_month_rate = last_month_metrics["daily_rate"]
    last_month_name = last_month_metrics["last_month_name"]
    last_month_trajectory = [round(float(last_month_rate), 2) for _ in range(len(future_exp))]

    # 4. Conditional Careless Line: ONLY if surge is active
    is_surge = False
    surge_z = 0.0
    driver_merchant = None
    driver_category = None
    careless_trajectory = None

    if surge_status:
        surge_z = float(surge_status.get("z_score", 0.0) or 0.0)
        surge_fired = bool(surge_status.get("alert_fired", False))
        driver_merchant = surge_status.get("top_merchant")
        driver_category = surge_status.get("top_category")
        if surge_fired or surge_z >= 2.0:
            is_surge = True

    if is_surge:
        today_spend = float(surge_status.get("today_spend", 0.0) or 0.0) if surge_status else 0.0
        careless_base = max(E0 * 1.5, today_spend if today_spend > 0 else E0 * 1.5)
        mu_careless = mu_E + 0.02
        careless_trajectory = [round(float(careless_base * math.exp(mu_careless * t)), 2) for t in range(len(future_exp))]

    # Future timeline dates
    now_date = datetime.now(timezone.utc).date()
    timeline = [(now_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(len(future_exp))]

    return {
        "timeline": timeline,
        "last_month_name": last_month_name,
        "rates": {
            "current_daily_rate": round(float(E0), 2),
            "optimal_daily_rate": round(float(optimal_rate), 2),
            "last_month_daily_rate": round(float(last_month_rate), 2)
        },
        "trajectories": {
            "expected": future_exp,
            "optimal": optimal_trajectory,
            "last_month": last_month_trajectory,
            "careless": careless_trajectory
        },
        "bounds": {
            "Mean": future_exp,
            "Optimal": optimal_trajectory,
            "LastMonth": last_month_trajectory,
            "Careless": careless_trajectory
        },
        "surge_status": {
            "is_active": is_surge,
            "z_score": round(surge_z, 2),
            "driver_merchant": driver_merchant,
            "driver_category": driver_category
        }
    }
