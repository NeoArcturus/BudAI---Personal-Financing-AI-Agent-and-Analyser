import os
import pandas as pd
import numpy as np
import logging
import torch
from sqlalchemy import text
from config import SessionLocal
from services.api_integrator.get_account_detail import UserAccounts
from services.Forecaster_Agent.mathematics.mathematics import run_hybrid_engine, run_converged_expense_engine
from services.Forecaster_Agent.models.parameter_lstm import ParameterLSTM
from models.database_models import ForecastParameters
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class ForecasterAgent:
    def __init__(self, db_path=None):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "lstm_params.pth")
        self.lstm = ParameterLSTM().to(self.device)
        if os.path.exists(self.model_path):
            self.lstm.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.lstm.eval()

    def fetch_live_balance(self, identifier, user_uuid):
        user_acc = UserAccounts(user_id=user_uuid)
        balance = user_acc.get_account_balance(identifier, user_uuid)
        return float(balance) if balance is not None else 0.0

    def get_user_params(self, user_uuid):
        with SessionLocal() as session:
            params = session.query(ForecastParameters).filter_by(user_uuid=user_uuid).first()
            if params:
                return {'kappa': params.kappa, 'theta': params.theta, 'xi': params.xi, 'rho': params.rho, 'lambda': params.lambda_val, 'mu_J': params.mu_j, 'sigma_j': params.sigma_j}
        return {'kappa': 2.0, 'theta': 0.04, 'xi': 0.1, 'rho': -0.5, 'lambda': 0.1, 'mu_J': -0.05, 'sigma_J': 0.1}

    def generate_dynamic_parameters(self, user_uuid):
        try:
            from services.memory_service import MemoryService
            mem = MemoryService()
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_transactions("ALL", user_uuid)
            if df.empty or len(df) < 30: return
            df['Date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
            df = df.sort_values('Date')
            amounts = df['amount'].values[-30:]
            cat_list = sorted(list(set(df['category'].dropna().unique())))
            cat_map = {cat: i for i, cat in enumerate(cat_list)}
            category_ids = df['category'].map(cat_map).fillna(-1).values[-30:]
            seasonal_context = mem.get_seasonal_context(user_uuid, limit=10)
            context_amounts = [m.get('amount', 0.0) for m in (seasonal_context['metadatas'][0] if seasonal_context['metadatas'] else [])]
            context_avg = np.mean(context_amounts) if context_amounts else 0.0
            feature_vector = np.array([amounts, category_ids, [context_avg]*30], dtype=np.float32).T
            tensor_in = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad(): out = self.lstm(tensor_in).cpu().numpy()[0]
            with SessionLocal() as session:
                params = session.query(ForecastParameters).filter_by(user_uuid=user_uuid).first()
                if not params:
                    params = ForecastParameters(user_uuid=user_uuid)
                    session.add(params)
                params.kappa, params.theta, params.xi, params.rho, params.lambda_val, params.mu_j, params.sigma_j = float(out[0]*5.0), float(out[1]*0.2), float(out[2]*0.5), float(out[3]*2.0-1.0), float(out[4]*0.5), float(out[5]*0.5-0.25), float(out[6]*0.3)
                session.commit()
        except Exception as e: logger.error(f"Dynamic params failed: {e}")

    def fetch_and_calculate_parameters(self, account_id, current_balance, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("SELECT b.bank_name FROM banks b LEFT JOIN accounts a ON b.bank_uuid = a.bank_uuid WHERE a.account_id=:account_id OR b.bank_name ILIKE :ident"), {"account_id": account_id, "ident": f"%{account_id}%"}).fetchone()
                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(identifier=bank_name, user_uuid=user_uuid)
        except Exception: df = pd.DataFrame()
        if df.empty: return current_balance, -0.01, 0.05
        df['Date'] = pd.to_datetime(df.get('date', df.get('timestamp')), format='ISO8601', utc=True).dt.date
        daily_net = df.groupby('Date')['amount'].sum().reset_index().sort_values('Date')
        daily_net['Reverse_Amount'] = daily_net['amount'].iloc[::-1]
        historical_balances, temp_balance = [current_balance], current_balance
        for amt in daily_net['Reverse_Amount'].values[:-1]:
            temp_balance -= amt
            historical_balances.append(temp_balance)
        daily_net['Balance'] = historical_balances[::-1]
        recent_data = daily_net.tail(lookback_days).copy()
        recent_data['Safe_Balance'] = recent_data['Balance'].apply(lambda x: max(x, 10))
        recent_data['Returns'] = np.log(recent_data['Safe_Balance'] / recent_data['Safe_Balance'].shift(1))
        mu = float(np.clip(recent_data['Returns'].mean(), -0.02, 0.02))
        sigma = recent_data['Returns'].std()
        return current_balance, (mu if not np.isnan(mu) else -0.01), (sigma if not np.isnan(sigma) and sigma != 0 else 0.05)

    def fetch_expense_parameters(self, account_id, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("SELECT b.bank_name FROM banks b LEFT JOIN accounts a ON b.bank_uuid = a.bank_uuid WHERE a.account_id=:account_id OR b.bank_name ILIKE :ident"), {"account_id": account_id, "ident": f"%{account_id}%"}).fetchone()
                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(identifier=bank_name, user_uuid=user_uuid)
        except Exception: df = pd.DataFrame()
        if df.empty: return 50.0, 0.001
        df['Date'] = pd.to_datetime(df.get('date', df.get('timestamp')), format='ISO8601', utc=True).dt.date
        df['amount'] = df['amount'].abs()
        recent_data = df.groupby('Date')['amount'].sum().reset_index().sort_values('Date').tail(lookback_days).copy()
        recent_data['Safe_Amount'] = recent_data['amount'].apply(lambda x: max(x, 1))
        E0 = recent_data['Safe_Amount'].mean()
        recent_data['Returns'] = np.log(recent_data['Safe_Amount'] / recent_data['Safe_Amount'].shift(1))
        mu_E = float(np.clip(recent_data['Returns'].mean(), -0.02, 0.02))
        return E0, (mu_E if not np.isnan(mu_E) else 0.001)

    def detect_upcoming_expenses(self, user_uuid, account_id=None, days=30, df_input=None):
        if df_input is not None:
            df = df_input
        else:
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_transactions(account_id, user_uuid)
            
        if df.empty: return np.zeros(days + 1), []
        
        df['Date'] = pd.to_datetime(df.get('date', df.get('timestamp')), format='ISO8601', utc=True)
        critical_categories = ["Bills & Utilities", "Housing", "Entertainment", "Transfers & Investments", "Taxes"]
        df_out = df[(df['amount'] < 0) & (df['category'].isin(critical_categories))].copy()
        
        calendar, timeline = np.zeros(days + 1), []
        now = pd.Timestamp.utcnow()
        
        for (desc, cat), group in df_out.groupby(['description', 'category']):
            if len(group) >= 2:
                group = group.sort_values('Date')
                deltas = group['Date'].diff().dt.days.dropna()
                median_delta = deltas.median()
                if 25 <= median_delta <= 35 and deltas.std() < 5:
                    last_date = group['Date'].iloc[-1]
                    next_date = last_date + pd.Timedelta(days=round(median_delta))
                    while (next_date - now).days <= days:
                        day_idx = (next_date - now).days
                        if 0 <= day_idx <= days:
                            calendar[day_idx] += group['amount'].median()
                            timeline.append({'day': day_idx, 'date': next_date.strftime('%Y-%m-%d'), 'merchant': desc, 'category': cat, 'amount': round(float(group['amount'].median()), 2)})
                        next_date += pd.Timedelta(days=round(median_delta))
        return calendar, sorted(timeline, key=lambda x: x['day'])

    def run_hybrid_simulation(self, account_id, S0, mu, user_uuid, days=60, paths=1000000, discipline_multiplier=1.0, drift_adjustment=0.0, stress_test_active=False, macro_environment="Stable"):
        params = self.get_user_params(user_uuid)
        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            if discipline_multiplier < 1.0: params['theta'], params['xi'] = params['theta']*0.8, params['xi']*0.7
            elif discipline_multiplier > 1.0: params['theta'], params['xi'] = params['theta']*1.4, params['xi']*1.5
            mu += drift_adjustment
            if macro_environment == "Inflationary": params['theta'], mu = params['theta']*1.2, mu-0.005
            elif macro_environment == "Recession": params['lambda'], params['mu_J'] = params['lambda']+0.2, params['mu_J']-0.1
            if stress_test_active: params['lambda'], params['mu_J'], params['sigma_j'] = 0.8, -0.4, 0.2
        calendar, timeline = self.detect_upcoming_expenses(user_uuid, account_id, days)
        df = run_hybrid_engine(S0, mu, params, days, paths, account_id, calendar)
        return df, timeline

    def run_expense_simulation(self, account_id, E0, mu, user_uuid, days=30, paths=1000000, discipline_multiplier=1.0, drift_adjustment=0.0, stress_test_active=False, macro_environment="Stable", current_balance=None):
        calendar, timeline = self.detect_upcoming_expenses(user_uuid, account_id, days)
        if np.all(calendar == 0):
            logger.info(f"No upcoming bills detected for {account_id}. Returning zero expense projection.")
            df = pd.DataFrame(np.zeros((1, days + 1)))
            return df, timeline
        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            if discipline_multiplier < 1.0: mu -= 0.01
            elif discipline_multiplier > 1.0: mu += 0.02
            mu += drift_adjustment
            if macro_environment == "Inflationary": mu += 0.015
            elif macro_environment == "Recession": mu -= 0.01
        df = run_converged_expense_engine(E0, mu, days, paths, account_id, calendar)
        if current_balance is not None and not df.empty:
            for row_idx in range(len(df)):
                path_vals = df.iloc[row_idx].values.copy()
                path_vals = np.minimum(path_vals, current_balance)
                df.iloc[row_idx] = path_vals
        return df, timeline
