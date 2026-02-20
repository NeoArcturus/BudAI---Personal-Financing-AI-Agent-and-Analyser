import re
import pandas as pd
from tqdm import tqdm


class Categorizer:
    def __init__(self):
        self.rules = {
            "Food & Dining": r"tesco|lidl|asda|greggs|pret|sainsbury|morrisons|aldi|mcdonalds|kfc|starbucks|costa",
            "Transportation": r"uber|trainline|tfl|bus|rail|stagecoach|bee network|taxi|bolt",
            "Bills & Utilities": r"lebara|vodafone|ee|o2|british gas|octopus|council tax|water|energy|mobile",
            "Shopping": r"amazon|ebay|argos|zara|h&m|ikea|currys",
            "Entertainment": r"netflix|spotify|cinema|vue|odeon|steam|playstation|xbox",
            "Health & Wellness": r"boots|pharmacy|gym|nhs|dentist",
            "Transfers & Investments": r"paypal|revolut|transfer|monzo|savings|investment"
        }

    def categorize_data(self, df):
        final_labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing Transactions"):
            text_lower = row['Description'].lower()
            amt = row['Amount']
            assigned = False

            if amt > 0:
                final_labels.append("Income")
                continue

            for category, pattern in self.rules.items():
                if re.search(pattern, text_lower):
                    final_labels.append(category)
                    assigned = True
                    break

            if not assigned:
                final_labels.append("Other")

        df.loc[:, "Category"] = final_labels
        return df
