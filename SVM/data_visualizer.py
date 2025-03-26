import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import numpy as np
import math


# ===================== SCATTER PLOT =====================
def scatter_plot(df, save_path):
    """Creates a scatter plot of Latitude and Longitude."""
    custom_palette = ["#E16540", "#9999FF", "#FAC666"]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x="LATITUDE",
        y="LONGITUDE",
        data=df,
        hue="ACCLASS",
        palette=custom_palette,
    )
    plt.title("Scatterplot of Latitude & Longitude", fontsize=14, fontweight="bold")
    plt.xlabel("Latitude", fontsize=12)
    plt.ylabel("Longitude", fontsize=12)
    plt.savefig(save_path)
    plt.show()


# ===================== PIE CHART =====================
def pie_chart(df, save_path):
    """Creates a pie chart for the target variable (ACCLASS)."""
    target_counts = df["ACCLASS"].value_counts()
    custom_colors = ["#95D1FF", "#FAC666", "#7661E2"]
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        target_counts,
        labels=target_counts.index,
        autopct="%1.1f%%",
        colors=custom_colors,
        wedgeprops={"edgecolor": "white"},
        startangle=140,
    )
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    plt.gca().add_artist(centre_circle)
    plt.title(
        "Distribution of Target Variable (ACCLASS)", fontsize=14, fontweight="bold"
    )
    plt.savefig(save_path)
    plt.show()


# ===================== BAR CHART =====================
def bar_chart(df, save_path):
    """Creates a bar chart for the ROAD_CLASS feature."""
    custom_bar_colors = [
        "#FAC666",
        "#7661E2",
        "#95D1FF",
        "#E16540",
        "#E3E0DE",
        "#FABE7A",
    ]
    plt.figure(figsize=(12, 6))
    sns.countplot(x="ROAD_CLASS", data=df, palette=custom_bar_colors)
    plt.title("Count of Road Class")
    plt.xticks(rotation=90)
    plt.savefig(save_path)
    plt.show()


# ===================== HEATMAP CORRELATION =====================
def heatmap_correlation(df, save_path):
    """Creates a heatmap for feature correlations."""
    custom_cmap = sns.color_palette("YlOrBr", as_cmap=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap=custom_cmap,
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.savefig(save_path)
    plt.show()


# ===================== HEATMAP MISSING VALUES =====================
def heatmap_missing_values(df, save_path):
    """Creates a heatmap for missing values in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap="Blues", cbar=False)
    plt.title("Missing Data Heatmap", fontsize=14, fontweight="bold")
    plt.savefig(save_path)
    plt.show()


# ===================== HISTOGRAM PLOT =====================
def hist_plot(df, save_path):
    """Creates histograms for numeric features."""
    df.hist(figsize=(12, 8), bins=30, color="#7661E2", edgecolor="black")
    plt.suptitle("Distribution of Numeric Features", fontsize=14)
    plt.savefig(save_path)
    plt.show()


# ===================== PAIR PLOT =====================
def pair_plot(df, save_path):
    """Creates a pair plot for numeric features."""
    custom_bar_colors = ["#FAC666", "#7661E2", "#95D1FF"]
    sns.pairplot(df, hue="ACCLASS", palette=custom_bar_colors)
    plt.title("Pairplot of Features")
    plt.savefig(save_path)
    plt.show()


# ===================== SPLINE PLOT =====================
def spline_plot(df, save_path):
    """Creates a spline plot for accident counts over time."""
    if "MONTH_YEAR" not in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["MONTH_YEAR"] = df["DATE"].dt.to_period("M").astype(str)

    # Group by 'MONTH_YEAR' and 'ACCLASS' to count occurrences
    accidents_by_month = (
        df.groupby(["MONTH_YEAR", "ACCLASS"])
        .size()
        .reset_index(name="COUNT")
    )

    # Convert 'MONTH_YEAR' to datetime for plotting
    accidents_by_month["MONTH_YEAR"] = pd.to_datetime(accidents_by_month["MONTH_YEAR"])

    # Define colors for each accident type
    colors = {
        "Fatal": "#F5866A",
        "Non-Fatal Injury": "#6956E5",
        "Property Damage O": "#59E6F6",
    }

    # Plot Spline Chart for each accident class
    plt.figure(figsize=(12, 6))

    for acclass in accidents_by_month["ACCLASS"].unique():
        subset = accidents_by_month[accidents_by_month["ACCLASS"] == acclass]
        x = subset["MONTH_YEAR"].astype(np.int64) // 10**9  # Convert datetime to timestamps
        y = subset["COUNT"]

        # Interpolation only if there are more than 2 data points
        if len(x) > 2:
            spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = spline(x_smooth)
            plt.plot(
                pd.to_datetime(x_smooth, unit="s"),
                y_smooth,
                color=colors.get(acclass, "black"),
                label=acclass,
            )

        # Scatter plot for actual data points
        plt.scatter(
            subset["MONTH_YEAR"],
            subset["COUNT"],
            color=colors.get(acclass, "black"),
            s=10,
        )

    # Labeling and styling the plot
    plt.xlabel("Year")
    plt.ylabel("Accident Count")
    plt.xticks(rotation=45)
    plt.title("Year vs. Accident Types", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(axis="y", linestyle="-", alpha=0.7)
    plt.savefig(save_path)
    plt.show()


# ===================== CATEGORICAL DISTRIBUTION =====================
def cat_distribution(df, save_path):
    """Plots the distribution of categorical features."""
    categorical_features = df.select_dtypes(exclude=["number"]).drop(columns=["DATE"])
    plots_per_figure = 12
    columns = 4
    rows = math.ceil(plots_per_figure / columns)

    for i in range(0, len(categorical_features.columns), plots_per_figure):
        fig, axes = plt.subplots(rows, columns, figsize=(18, 14))
        fig.suptitle("Categorical Feature Distributions", fontsize=18)

        for j, column in enumerate(
            categorical_features.columns[i : i + plots_per_figure]
        ):
            ax = axes[j // columns, j % columns]
            categorical_features[column].value_counts().plot(
                kind="bar", color="#7661E2", edgecolor="black", ax=ax
            )
            ax.set_title(f"Distribution of {column}", fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=45)

        for k in range(j + 1, rows * columns):
            fig.delaxes(axes.flatten()[k])

        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.show()


# ===================== CREATE FULL CORRELATION MATRICES =====================
def create_full_correlation_matrices(df):
    """Create and save correlation matrices for all columns, including categorical ones"""
    print("\n=== Creating Full Correlation Matrices (Including Categorical) ===")
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Remove 'Property Damage O' records as they're not relevant for severity analysis
    df_encoded = df_encoded[df_encoded['ACCLASS'] != 'Property Damage O']
    
    # Dictionary to store original categories for each column
    category_mappings = {}
    
    # Ensure 'DATE' column is in datetime format
    df_encoded["DATE"] = pd.to_datetime(df_encoded["DATE"], errors='coerce')

    # Check if there are any invalid dates
    if df_encoded["DATE"].isna().sum() > 0:
        print(f"Warning: {df_encoded['DATE'].isna().sum()} rows have invalid or missing date values.")

    # Extract Month and Day of the Week
    df_encoded['Month'] = df_encoded['DATE'].dt.month
    df_encoded['DayOfWeek'] = df_encoded['DATE'].dt.dayofweek

    # Drop the original 'DATE' column
    df_encoded = df_encoded.drop('DATE', axis=1)

    # Add target column first (before encoding other categoricals)
    df_encoded['ACCLASS'] = (df_encoded['ACCLASS'] == 'Fatal').astype(float)
    
    # Drop string columns that shouldn't be part of correlation
    columns_to_drop = ['STREET1', 'STREET2', 'OFFSET']
    df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')
    
    # Encode all remaining categorical columns while preserving original names
    for column in df_encoded.select_dtypes(include=['object']).columns:
        # For binary columns (Yes/No)
        if set(df_encoded[column].dropna().unique()).issubset({'Yes', 'No'}):
            df_encoded[column] = (df_encoded[column] == 'Yes').astype(float)
        # For other categorical columns
        else:
            # Store original categories
            unique_values = df_encoded[column].dropna().unique()
            category_mappings[column] = {
                idx: cat for idx, cat in enumerate(unique_values)
            }
            # Use label encoding
            df_encoded[column] = pd.Categorical(df_encoded[column]).codes.astype(float)
    
    # Calculate correlations
    correlations = df_encoded.corr()
    
    # Save category mappings for reference
    with open('./insights/correlation/category_mappings.txt', 'w') as f:
        f.write("Category Mappings for Encoded Features:\n")
        f.write("=====================================\n\n")
        for column, mapping in category_mappings.items():
            f.write(f"\n{column}:\n")
            for code, category in mapping.items():
                f.write(f"  {code}: {category}\n")
    
    # Save correlations to CSV
    correlations.to_csv('./insights/correlation/full_feature_correlations.csv')
    
    # Create correlation plots
    n_cols = len(correlations.columns)
    max_cols_per_plot = 20
    n_splits = (n_cols + max_cols_per_plot - 1) // max_cols_per_plot
    
    for i in range(n_splits):
        start_idx = i * max_cols_per_plot
        end_idx = min((i + 1) * max_cols_per_plot, n_cols)
        
        plt.figure(figsize=(24, 20))
        sns.heatmap(
            correlations.iloc[start_idx:end_idx, start_idx:end_idx],
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True
        )
        plt.title(f'Full Correlation Matrix Part {i+1}')
        plt.tight_layout()
        plt.savefig(f'./insights/correlation/full_correlation_matrix_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create specific correlation plot with target
    plt.figure(figsize=(15, 10))
    target_corr = correlations['ACCLASS'].sort_values(ascending=False)
    target_corr = target_corr.drop('ACCLASS')
    
    # Plot top 30 correlations
    top_30_corr = target_corr.head(30)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_30_corr.values, y=top_30_corr.index)
    plt.title('Top 30 Feature Correlations with Accidents')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('./insights/correlation/full_target_correlations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return target_corr, category_mappings