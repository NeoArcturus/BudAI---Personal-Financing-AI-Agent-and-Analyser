import matplotlib.pyplot as mp
import seaborn as sb
import os


class CategoryDistribution:
    def __init__(self, classified_data):
        self.classified_data = classified_data
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.base_dir, '..'))
        self.img_dir = os.path.join(self.root_dir, "saved_media", "images")
        os.makedirs(self.img_dir, exist_ok=True)

    def pie_chart_category_distribution(self):
        category_counts = self.classified_data['Category'].value_counts()
        mp.figure(figsize=(8, 8))
        mp.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               startangle=140, colors=sb.color_palette('viridis', len(category_counts)))
        mp.title('Expense Distribution by Category')
        mp.axis('equal')
        mp.savefig(os.path.join(
            self.img_dir, "category_distribution_pie_chart.png"))
        mp.close()

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
        mp.savefig(os.path.join(
            self.img_dir, 'category_distribution_bar_plot.png'))
        mp.close()
