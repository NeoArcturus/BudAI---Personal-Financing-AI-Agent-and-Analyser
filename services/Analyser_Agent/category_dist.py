import pandas as pd
import logging
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class CategoryDistribution:
    def __init__(self, classified_data):
        self.classified_data = classified_data
    def get_category_distribution_data(self):
        df = self.classified_data.copy()
        df['Amount'] = df['Amount'].abs()
        grouped = df.groupby('Category')['Amount'].sum().reset_index()
        grouped.columns = ['Category', 'Total_Amount']
        grouped = grouped.sort_values(by='Total_Amount', ascending=False)
        return grouped

