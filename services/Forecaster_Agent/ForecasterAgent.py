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
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.path.dirname(
            __file__), "models", "lstm_params.pth")
        self.lstm = ParameterLSTM().to(self.device)
        if os.path.exists(self.model_path):
            self.lstm.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
        self.lstm.eval()

    def fetch_live_balance(self, identifier, user_uuid):
        if str(identifier).upper() == "ALL" or "," in identifier:
            raise ValueError(
                "ForecasterAgent strictly handles a single account identifier.")
        user_acc = UserAccounts(user_id=user_uuid)
        balance = user_acc.get_account_balance(identifier, user_uuid)
        return float(balance) if balance is not None else 0.0

    def get_user_params(self, user_uuid):
        with SessionLocal() as session:
            params = session.query(ForecastParameters).filter_by(
                user_uuid=user_uuid).first()
            if params:
                return {
                    'kappa': params.kappa,
                    'theta': params.theta,
                    'xi': params.xi,
                    'rho': params.rho,
                    'lambda': params.lambda_val,
                    'mu_J': params.mu_j,
                    'sigma_j': params.sigma_j
                }
        return {
            'kappa': 2.0,
            'theta': 0.04,
            'xi': 0.1,
            'rho': -0.5,
            'lambda': 0.1,
            'mu_J': -0.05,
            'sigma_J': 0.1
        }

    def generate_dynamic_parameters(self, user_uuid):
        try:
            from services.memory_service import MemoryService
            mem = MemoryService()
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_transactions("ALL", user_uuid)
            if df.empty or len(df) < 30:
                return
            df['Date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
            df = df.sort_values('Date')
            amounts = df['amount'].values[-30:]
            cat_list = sorted(list(set(df['category'].dropna().unique())))
            cat_map = {cat: i for i, cat in enumerate(cat_list)}
            category_ids = df['category'].map(cat_map).fillna(-1).values[-30:]
            seasonal_context = mem.get_seasonal_context(user_uuid, limit=10)
            context_amounts = [m.get('amount', 0.0) for m in (
                seasonal_context['metadatas'][0] if seasonal_context['metadatas'] else [])]
            context_avg = np.mean(context_amounts) if context_amounts else 0.0
            feature_vector = np.array(
                [amounts, category_ids, [context_avg]*30], dtype=np.float32).T
            tensor_in = torch.tensor(
                feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            logger.info(f"LSTM Input Tensor Shape: {tensor_in.shape}")
            logger.info(
                f"LSTM Sample Input Data (First 3 days) - [Amount, CatID, SeasonalBias]:\n{feature_vector[:3]}")
            with torch.no_grad():
                out = self.lstm(tensor_in).cpu().numpy()[0]
            with SessionLocal() as session:
                params = session.query(ForecastParameters).filter_by(
                    user_uuid=user_uuid).first()
                if not params:
                    params = ForecastParameters(user_uuid=user_uuid)
                    session.add(params)
                params.kappa = float(out[0] * 5.0)
                params.theta = float(out[1] * 0.2)
                params.xi = float(out[2] * 0.5)
                params.rho = float(out[3] * 2.0 - 1.0)
                params.lambda_val = float(out[4] * 0.5)
                params.mu_j = float(out[5] * 0.5 - 0.25)
                params.sigma_j = float(out[6] * 0.3)
                session.commit()
                logger.info(
                    f"LSTM Generated Bates Parameters for {user_uuid}: kappa={params.kappa:.4f}, theta={params.theta:.4f}, xi={params.xi:.4f}, lambda={params.lambda_val:.4f}")
        except Exception as e:
            logger.error("An error occurred in this block", exc_info=True)
            logger.error(f"Failed to generate dynamic parameters: {e}")

    def fetch_and_calculate_parameters(self, account_id, current_balance, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("""SELECT b.bank_name
                               FROM banks b
                               LEFT JOIN accounts a
                               ON b.bank_uuid = a.bank_uuid
                               WHERE a.account_id=:account_id"""), {"account_id": account_id}).fetchone()
                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(
                    identifier=bank_name, user_uuid=user_uuid)
        except Exception:
            logger.error("An error occurred in this block", exc_info=True)
            df = pd.DataFrame()
        if df.empty:
            return current_balance, -0.01, 0.05
        if 'timestamp' in df.columns and 'date' not in df.columns:
            df['date'] = df['timestamp']
        df['Date'] = pd.to_datetime(
            df['date'], format='ISO8601', utc=True).dt.date
        daily_net = df.groupby('Date')['amount'].sum(
        ).reset_index().sort_values('Date')
        daily_net['Reverse_Amount'] = daily_net['amount'].iloc[::-1]
        historical_balances = [current_balance]
        temp_balance = current_balance
        for amt in daily_net['Reverse_Amount'].values[:-1]:
            temp_balance -= amt
            historical_balances.append(temp_balance)
        daily_net['Balance'] = historical_balances[::-1]
        recent_data = daily_net.tail(lookback_days).copy()
        recent_data['Safe_Balance'] = recent_data['Balance'].apply(
            lambda x: max(x, 10))
        recent_data['Returns'] = np.log(
            recent_data['Safe_Balance'] / recent_data['Safe_Balance'].shift(1))
        mu = recent_data['Returns'].mean()
        mu = float(np.clip(mu, -0.02, 0.02))
        sigma = recent_data['Returns'].std()
        if np.isnan(mu):
            mu = -0.01
        if np.isnan(sigma) or sigma == 0:
            sigma = 0.05
        return current_balance, mu, sigma

    def fetch_expense_parameters(self, account_id, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("""SELECT b.bank_name
                               FROM banks b
                               LEFT JOIN accounts a
                               ON b.bank_uuid = a.bank_uuid
                               WHERE a.account_id=:account_id"""), {"account_id": account_id}).fetchone()
                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(
                    identifier=bank_name, user_uuid=user_uuid)
        except Exception:
            logger.error("An error occurred in this block", exc_info=True)
            df = pd.DataFrame()
        if df.empty:
            return 50.0, 0.001
        if 'timestamp' in df.columns and 'date' not in df.columns:
            df['date'] = df['timestamp']
        df['Date'] = pd.to_datetime(
            df['date'], format='ISO8601', utc=True).dt.date
        df['amount'] = df['amount'].abs()
        daily_expenses = df.groupby(
            'Date')['amount'].sum().reset_index().sort_values('Date')
        recent_data = daily_expenses.tail(lookback_days).copy()
        recent_data['Safe_Amount'] = recent_data['amount'].apply(
            lambda x: max(x, 1))
        E0 = recent_data['Safe_Amount'].mean()
        recent_data['Returns'] = np.log(
            recent_data['Safe_Amount'] / recent_data['Safe_Amount'].shift(1))
        mu_E = recent_data['Returns'].mean()
        if np.isnan(mu_E):
            mu_E = 0.001
        mu_E = float(np.clip(mu_E, -0.02, 0.02))
        return E0, mu_E

    def detect_upcoming_expenses(self, user_uuid, df_input=None):
        if df_input is not None:
            df = df_input.copy()
        else:
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_transactions("ALL", user_uuid)
            
        if df.empty:
            return []
        
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        df_out = df[df['amount'] < 0].copy()
        upcoming = []
        now = pd.Timestamp.utcnow()
        for desc, group in df_out.groupby('description'):
            if len(group) >= 2:
                group = group.sort_values('Date')
                deltas = group['Date'].diff().dt.days.dropna()
                median_delta = deltas.median()
                if 25 <= median_delta <= 35 and deltas.std() < 5:
                    last_date = group['Date'].iloc[-1]
                    next_date = last_date + \
                        pd.Timedelta(days=round(median_delta))
                    days_until = (next_date - now).days
                    if 0 <= days_until <= 30:
                        upcoming.append({
                            'merchant': desc,
                            'amount': group['amount'].median(),
                            'next_date': next_date.strftime('%Y-%m-%d'),
                            'days_until': days_until
                        })
        return sorted(upcoming, key=lambda x: x['days_until'])

    def run_hybrid_simulation(self, account_id, S0, mu, user_uuid, days=60, paths=1000,
                              discipline_multiplier=1.0, drift_adjustment=0.0,
                              stress_test_active=False, macro_environment="Stable"):
        params = self.get_user_params(user_uuid)
        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            logger.warning(
                f"SIMULATION OVERRIDES APPLIED for user {user_uuid} on account {account_id}")
            if discipline_multiplier < 1.0:
                params['theta'] *= 0.8
                params['xi'] *= 0.7
            elif discipline_multiplier > 1.0:
                params['theta'] *= 1.4
                params['xi'] *= 1.5
            mu += drift_adjustment
            if macro_environment == "Inflationary":
                params['theta'] *= 1.2
                mu -= 0.005
            elif macro_environment == "Recession":
                params['lambda'] += 0.2
                params['mu_J'] -= 0.1
            if stress_test_active:
                params['lambda'] = 0.8
                params['mu_J'] = -0.4
                params['sigma_j'] = 0.2
        df = run_hybrid_engine(S0, mu, params, days, paths, account_id)
        upcoming = self.detect_upcoming_expenses(user_uuid)
        for row_idx in range(len(df)):
            path_vals = df.iloc[row_idx].values.copy()
            for b in upcoming:
                day_idx = b['days_until']
                if 0 < day_idx <= days:
                    path_vals[day_idx:] += b['amount']
            path_vals = np.maximum(path_vals, 0.0)
            df.iloc[row_idx] = path_vals
        return df

    def run_expense_simulation(self, account_id, E0, mu, days=30, paths=1000,
                               discipline_multiplier=1.0, drift_adjustment=0.0,
                               stress_test_active=False, macro_environment="Stable",
                               current_balance=None):
        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            if discipline_multiplier < 1.0:
                mu -= 0.01
            elif discipline_multiplier > 1.0:
                mu += 0.02
            mu += drift_adjustment
            if macro_environment == "Inflationary":
                mu += 0.015
            elif macro_environment == "Recession":
                mu -= 0.01
        df = run_converged_expense_engine(E0, mu, days, paths, account_id)
        if current_balance is not None and not df.empty:
            for row_idx in range(len(df)):
                path_vals = df.iloc[row_idx].values.copy()
                path_vals = np.minimum(path_vals, current_balance)
                df.iloc[row_idx] = path_vals
        return df
