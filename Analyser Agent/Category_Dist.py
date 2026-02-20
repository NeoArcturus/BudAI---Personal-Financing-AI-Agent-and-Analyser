import matplotlib.pyplot as mp
import seaborn as sb


class CategoryDistribution:
    def __init__(self, classified_data):
        self.classified_data = classified_data

    def plot_category_distribution(self):
        category_counts = self.classified_data['Category'].value_counts()
        mp.figure(figsize=(10, 6))
        sb.barplot(x=category_counts.index,
                   y=category_counts.values, palette='viridis')
        mp.title('Distribution of Expenses by Category')
        mp.xlabel('Category')
        mp.ylabel('Count')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.show()

    def pie_chart_category_distribution(self):
        category_counts = self.classified_data['Category'].value_counts()
        mp.figure(figsize=(8, 8))
        mp.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               startangle=140, colors=sb.color_palette('viridis', len(category_counts)))
        mp.title('Expense Distribution by Category')
        mp.axis('equal')
        mp.show()

    def save_bar_plot(self):
        category_counts = self.classified_data['Category'].value_counts()
        mp.figure(figsize=(10, 6))
        sb.barplot(x=category_counts.index,
                   y=category_counts.values, palette='viridis')
        mp.title('Distribution of Expenses by Category')
        mp.xlabel('Category')
        mp.ylabel('Count')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.savefig('category_distribution_bar_plot.png')
