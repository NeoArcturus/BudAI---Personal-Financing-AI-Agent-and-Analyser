import pandas as pd
from datetime import datetime, timedelta
from services.api_integrator.get_account_detail import UserAccounts
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent

class ProfileBuilder:
    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid

    def build_profile(self) -> str:
        user_acc = UserAccounts(user_id=self.user_uuid)
        accounts = user_acc.get_all_accounts()
        total_balance = sum(acc['balance'] for acc in accounts if acc['status'] == 'active')
        
        df = user_acc.get_transactions("ALL", self.user_uuid)
        
        if df.empty:
            return f"[Live Balance: £{total_balance:.2f} | No historical data]"
            
        df['Date'] = pd.to_datetime(df['date'], utc=True).dt.date
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        ninety_days_ago = datetime.utcnow().date() - timedelta(days=90)
        df_90 = df[df['Date'] >= ninety_days_ago]
        
        inflow = df_90[df_90['amount'] > 0]['amount'].sum()
        outflow = df_90[df_90['amount'] < 0]['amount'].sum()
        net_cash_flow = inflow + outflow
        
        forecaster = ForecasterAgent()
        upcoming_bills = forecaster.detect_upcoming_expenses(self.user_uuid)
        upcoming_str = ", ".join([f"{b['merchant']}: £{abs(b['amount']):.2f} ({b['days_until']} days)" for b in upcoming_bills])
        
        df_out = df[df['amount'] < 0].copy()
        df_out['abs_amount'] = df_out['amount'].abs()
        top_merchants_vol = df_out.groupby('description')['abs_amount'].sum().nlargest(5)
        top_merchants_freq = df_out.groupby('description').size().nlargest(5)
        
        vol_str = ", ".join([f"{k}: £{v:.2f}" for k, v in top_merchants_vol.items()])
        freq_str = ", ".join([f"{k}: {v}x" for k, v in top_merchants_freq.items()])
        
        from services.memory_service import MemoryService
        try:
            mem = MemoryService()
            seasonal_context = mem.get_seasonal_context(self.user_uuid, limit=3)
            context_docs = (seasonal_context['documents'][0] if seasonal_context['documents'] else [])
            rag_str = " | ".join(context_docs) if context_docs else "No historical seasonal parallels found."
        except Exception:
            rag_str = "Semantic Memory currently unavailable."
        
        profile = f"""
        [Tier 1 - Liquidity]
        Live Total Balance: £{total_balance:.2f}
        90-Day Net Cash Flow: £{net_cash_flow:.2f} (In: £{inflow:.2f}, Out: £{abs(outflow):.2f})

        [Tier 2 - Rhythm (Upcoming Bills)]
        {upcoming_str if upcoming_str else "No deterministic upcoming bills detected."}

        [Tier 3 - Footprint (Top Merchants)]
        By Volume: {vol_str}
        By Frequency: {freq_str}

        [Tier 4 - Semantic Memory]
        {rag_str}
        """
        return profile.strip()
