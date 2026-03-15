import pandas as pd
import os


class CategoryDistribution:
    def __init__(self, classified_data):
        self.classified_data = classified_data
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(
            os.path.join(self.base_dir, '..', '..'))
        self.csv_dir = os.path.join(self.root_dir, "saved_media", "csvs")
        os.makedirs(self.csv_dir, exist_ok=True)

    def extract_category_distribution_data(self, suffix):
        df = self.classified_data.copy()
        df['Amount'] = df['Amount'].abs()

        grouped = df.groupby('Category')['Amount'].sum().reset_index()
        grouped.columns = ['Category', 'Total_Amount']

        grouped = grouped.sort_values(by='Total_Amount', ascending=False)

        csv_path = os.path.join(
            self.csv_dir, f"total_per_category_{suffix}.csv")
        grouped.to_csv(csv_path, index=False)

        return csv_path
