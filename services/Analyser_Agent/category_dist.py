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

    def extract_category_distribution_data(self, account_id):
        category_counts = self.classified_data['Category'].value_counts(
        ).reset_index()
        category_counts.columns = ['Category', 'Count']
        csv_path = os.path.join(
            self.csv_dir, f"category_distribution_{account_id}.csv")
        category_counts.to_csv(csv_path, index=False)
        return csv_path
