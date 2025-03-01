import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import numpy as np
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
        custom_colors = ["#95D1FF", "#FAC666", "#7661E2"]
        plt.figure(figsize=(7, 7))
        wedges, texts, autotexts = plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=custom_colors, wedgeprops={'edgecolor': 'white'}, startangle=140)
        centre_circle = plt.Circle((0, 0), 0.60, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title("Distribution of Target Variable (ACCLASS)", fontsize=14, fontweight="bold")
        plt.savefig(save_path)
        plt.show()

    def bar_chart(self, save_path):
        custom_bar_colors = ["#FAC666", "#7661E2", "#95D1FF", "#E16540", "#E3E0DE","#FAC666", "#7661E2", "#95D1FF", "#E16540", "#E3E0DE","#FABE7A"]
        plt.figure(figsize=(12, 6))
        sns.countplot(x='ROAD_CLASS', data=self.data_ksi, palette=custom_bar_colors)
        plt.title("Count of Road Class")
        plt.xticks(rotation=90)
        plt.savefig(save_path)
        plt.show()

    def heatmap_correlation(self, save_path):
        custom_cmap = sns.color_palette("YlOrBr", as_cmap=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data_ksi.corr(numeric_only=True), annot=True, cmap=custom_cmap, fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.savefig(save_path)
        plt.show()

    def heatmap_missing_values(self,save_path):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data_ksi.isnull(), cmap="Blues", cbar=False)
        plt.title("Missing Data Heatmap", fontsize=14, fontweight='bold')
        plt.savefig(save_path)
        plt.show()

    def hist_plot(self, save_path):
        self.data_ksi.hist(figsize=(12, 8), bins=30, color="#7661E2", edgecolor="black")
        plt.suptitle("Distribution of Numeric Features", fontsize=14)
        plt.savefig(save_path)
        plt.show()

    def pair_plot(self, save_path):
        custom_bar_colors = ["#FAC666", "#7661E2", "#95D1FF"]
        sns.pairplot(self.data_ksi, hue="ACCLASS", palette=custom_bar_colors)
        plt.title("Pairplot of Features")
        plt.savefig(save_path)
        plt.show()
  
    def spline_plot(self, save_path):
        # Ensure the 'DATE' column is in datetime format
        self.data_ksi['DATE'] = pd.to_datetime(self.data_ksi['DATE'])

        # Create 'Month-Year' column
        self.data_ksi['MONTH_YEAR'] = self.data_ksi['DATE'].dt.to_period('M').astype(str)

        # Group by 'MONTH_YEAR' and 'ACCLASS' to count occurrences
        accidents_by_month = self.data_ksi.groupby(['MONTH_YEAR', 'ACCLASS']).size().reset_index(name='COUNT')

        # Convert 'MONTH_YEAR' to datetime for plotting
        accidents_by_month['MONTH_YEAR'] = pd.to_datetime(accidents_by_month['MONTH_YEAR'])

        # Define colors for each accident type
        colors = {
            'Fatal': '#F5866A',
            'Non-Fatal Injury': '#6956E5',
            'Property Damage O': '#59E6F6'
        }

        # Plot Spline Chart for each accident class
        plt.figure(figsize=(12, 6))

        for acclass in accidents_by_month['ACCLASS'].unique():
            subset = accidents_by_month[accidents_by_month['ACCLASS'] == acclass]
            x = subset['MONTH_YEAR'].astype(np.int64) // 10**9  # Convert datetime to timestamps
            y = subset['COUNT']

            # Interpolation only if there are more than 2 data points
            if len(x) > 2:
                spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = spline(x_smooth)
                plt.plot(pd.to_datetime(x_smooth, unit='s'), y_smooth, color=colors.get(acclass, 'black'), label=acclass)

            # Scatter plot for actual data points
            plt.scatter(subset['MONTH_YEAR'], subset['COUNT'], color=colors.get(acclass, 'black'), s=10)

        # Labeling and styling the plot
        plt.xlabel("Year")
        plt.ylabel("Accident Count")
        plt.xticks(rotation=45)

        plt.suptitle("Year vs. Accident Types", x=0.0, ha='left', fontweight='bold')
        plt.subplots_adjust(top=0.85)

        # Adjust legend placement
        plt.legend(
            loc='upper right', 
            bbox_to_anchor=(1, 1.15),  # Move it to the right
            ncol=3,  
            frameon=False
        )

        # Remove borders (spines) except the bottom one
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Show only horizontal grid lines
        plt.grid(axis='y', linestyle='-', alpha=0.7)

        # Save the plot
        plt.savefig(save_path)
        plt.show()