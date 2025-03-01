import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class Visualizer:
    def __init__(self, data_ksi):
        self.data_ksi = data_ksi


    def scatter_plot(self, save_path):
        custom_palette = ["#E16540", "#9999FF", "#FAC666"]
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='LATITUDE', y='LONGITUDE', data=self.data_ksi, hue='ACCLASS', palette=custom_palette)
        plt.title("Scatterplot of Latitude & Longitude", fontsize=14, fontweight='bold')
        plt.xlabel("Latitude", fontsize=12)
        plt.ylabel("Longitude", fontsize=12)
        plt.savefig(save_path)
        plt.show()

    def pie_chart(self, save_path):
        target_counts = self.data_ksi["ACCLASS"].value_counts()
        custom_colors = ["#95D1FF", "#FAC666", "#7661E2", "#E16540", "#E3E0DE"]
        plt.figure(figsize=(7, 7))
        wedges, texts, autotexts = plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=custom_colors, wedgeprops={'edgecolor': 'white'}, startangle=140)
        centre_circle = plt.Circle((0, 0), 0.60, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title("Distribution of Target Variable (ACCLASS)", fontsize=14, fontweight="bold")
        plt.savefig(save_path)
        plt.show()

    def bar_chart(self, save_path):
        custom_bar_colors = ["#FAC666", "#7661E2", "#95D1FF", "#E16540", "#E3E0DE"]
        plt.figure(figsize=(12, 6))
        sns.countplot(x='ROAD_CLASS', data=self.data_ksi, palette=custom_bar_colors)
        plt.title("Count of Road Class")
        plt.xticks(rotation=90)
        plt.savefig(save_path)
        plt.show()


