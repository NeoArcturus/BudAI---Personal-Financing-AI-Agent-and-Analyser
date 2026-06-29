import pandas as pd
import os
import asyncio
from datetime import datetime, timedelta
from services.api_integrator.get_account_detail import UserAccounts
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge

logger = get_core_logger("profile_builder")


class ProfileBuilder:
    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid

    async def build_profile(self) -> str:
        logger.info(f"Building profile for user {self.user_uuid}")
        try:
            user_acc = UserAccounts(user_id=self.user_uuid)
            
            logger.debug("Fetching recent transactions (180d window)")
            now_dt = datetime.now()
            from_date = (now_dt - timedelta(days=180)).strftime("%Y-%m-%d")
            to_date = now_dt.strftime("%Y-%m-%d")
            
            df = await asyncio.to_thread(user_acc.get_transactions, "ALL", self.user_uuid, from_date, to_date)
            
            logger.debug("Fetching all accounts for balance")
            accounts = await asyncio.to_thread(user_acc.get_all_accounts, skip_sync=True)
            active_accounts = [acc for acc in accounts if acc.get('status') == 'active']
            total_balance = sum(acc.get('balance', 0.0) for acc in active_accounts)

            if df.empty:
                logger.info("No transaction data found for user")
                return f"[Live Balance: £{total_balance:.2f} | No historical data]"

            logger.debug("Processing transaction data")
            df['Date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True).dt.date
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

            ninety_days_ago = (now_dt - timedelta(days=90)).date()
            df_90 = df[df['Date'] >= ninety_days_ago]

            inflow = df_90[df_90['amount'] > 0]['amount'].sum()
            outflow = df_90[df_90['amount'] < 0]['amount'].sum()
            net_cash_flow = inflow + outflow

            logger.debug("Detecting upcoming expenses (Zero DB hit)")
            forecaster = ForecasterAgent()
            _, timeline = forecaster.detect_upcoming_expenses(self.user_uuid, df_input=df)
            upcoming_str = ", ".join(
                [f"{b['merchant']}: £{abs(b['amount']):.2f} (in {b['day']} days)" for b in timeline])

            df_out = df[df['amount'] < 0].copy()
            df_out['abs_amount'] = df_out['amount'].abs()

            top_merchants_vol = df_out.groupby('description')['abs_amount'].sum().nlargest(5)
            top_categories = df_out.groupby('category')['abs_amount'].sum().nlargest(5)

            vol_str = ", ".join([f"{k}: £{v:.2f}" for k, v in top_merchants_vol.items()])
            cat_str = ", ".join([f"{k}: £{v:.2f}" for k, v in top_categories.items()])

            logger.debug("Offloading semantic memory lookup to microservice")
            bridge = MCPBridge()
            try:
                rag_str = await bridge.call_tool(
                    "memory",
                    "get_seasonal_behavior_context",
                    {"user_uuid": self.user_uuid}
                )
            except Exception as e:
                logger.warning(f"Microservice Memory access failed: {e}")
                rag_str = "Behavioral context temporarily unavailable."
            profile = f"""
            [Tier 1 - Liquidity]
            Live Total Balance: £{total_balance:.2f}
            90-Day Net Cash Flow: £{net_cash_flow:.2f} (In: £{inflow:.2f}, Out: £{abs(outflow):.2f})

            [Tier 2 - Rhythm (Upcoming Bills)]
            {upcoming_str if upcoming_str else "No deterministic upcoming bills detected."}

            [Tier 3 - Footprint (Merchant & Category Intelligence)]
            Top Categories: {cat_str}
            Top Merchants: {vol_str}

            [Tier 4 - Semantic Memory & Behavioral Trends]
            {rag_str}
            """
            return profile.strip()

        except Exception as e:
            logger.error(f"Failed to build profile for {self.user_uuid}: {e}", exc_info=True)
            return f"Error building profile: {str(e)}"
