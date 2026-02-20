import matplotlib.pyplot as mp
import pandas as pd


class ExpenseAnalysis:
    def __init__(self, classified_data):
        self.classified_data = classified_data

    def plot_weekly_spend(self):
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        weekly_spend = self.classified_data['Amount'].resample('W').sum()
        mp.figure(figsize=(10, 6))
        mp.plot(weekly_spend.index, weekly_spend.values, marker='o')
        mp.title('Weekly Spend Over Time')
        mp.xlabel('Week')
        mp.ylabel('Total Spend')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.show()

    def plot_monthly_spend(self):
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        monthly_spend = self.classified_data['Amount'].resample('M').sum()
        mp.figure(figsize=(10, 6))
        mp.plot(monthly_spend.index, monthly_spend.values, marker='o')
        mp.title('Monthly Spend Over Time')
        mp.xlabel('Month')
        mp.ylabel('Total Spend')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.show()
