import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def _fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def generate_visualizations(csv_file_stream):
    df = pd.read_csv(csv_file_stream)
    df.columns = df.columns.str.lower().str.strip()

    num_cols = ['age', 'tenure_months', 'total_rides', 'average_ride_cost',
                'days_since_last_ride', 'cancellation_rate',
                'customer_satisfaction_score', 'app_usage_hours_per_week']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    churn_rate = 0.0
    retention_rate = 0.0
    if 'churn_status' in df.columns:
        mapped = df['churn_status'].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0})
        mapped = pd.to_numeric(mapped, errors='coerce').fillna(0).astype(int)
        counts = mapped.value_counts()
        total = len(mapped)
        if total > 0:
            churn_rate = float((counts.get(1, 0) / total) * 100)
            retention_rate = float((counts.get(0, 0) / total) * 100)

    visualizations = {}

    # 1. Descriptive Analysis
    desc_html = df.describe().to_html(classes="min-w-full text-sm text-left text-gray-700", border=0)
    visualizations['Descriptive Analysis'] = {'type': 'html', 'content': desc_html}

    # 2. Distribution Analysis
    plot_cols = [c for c in ['age', 'total_rides', 'average_ride_cost', 'customer_satisfaction_score'] if c in df.columns]
    if plot_cols:
        fig, axes = plt.subplots(1, len(plot_cols), figsize=(15, 5))
        if len(plot_cols) == 1: axes = [axes]
        for idx, col in enumerate(plot_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='dodgerblue')
            axes[idx].set_title(f'Distribution of {col}')
        plt.tight_layout()
        visualizations['Distribution Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 3. Correlation Analysis
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty and len(num_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        visualizations['Correlation Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 4. Missing Value Analysis
    missing = df.isnull().sum()
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing = missing[missing > 0]
        sns.barplot(x=missing.index, y=missing.values, ax=ax, color='steelblue')
        plt.title('Missing Values per Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['Missing Value Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}
    else:
        visualizations['Missing Value Analysis'] = {'type': 'text', 'content': 'No missing values found in the dataset.'}

    # 5. Outlier Analysis
    outlier_cols = [c for c in ['age', 'average_ride_cost', 'days_since_last_ride', 'cancellation_rate'] if c in df.columns]
    if outlier_cols:
        fig, axes = plt.subplots(1, len(outlier_cols), figsize=(15, 5))
        if len(outlier_cols) == 1: axes = [axes]
        for idx, col in enumerate(outlier_cols):
            sns.boxplot(y=df[col].dropna(), ax=axes[idx], color='lightskyblue')
            axes[idx].set_title(f'Outliers in {col}')
        plt.tight_layout()
        visualizations['Outlier Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 6. Categorical Analysis
    cat_cols = [c for c in ['churn_status', 'gender', 'city', 'membership_type', 'payment_method'] if c in df.columns]
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(20, 5))
        if len(cat_cols) == 1: axes = [axes]
        for idx, col in enumerate(cat_cols):
            sns.countplot(x=df[col].dropna(), ax=axes[idx], hue=df[col].dropna(), palette='Blues', legend=False)
            axes[idx].set_title(f'Count of {col}')
            axes[idx].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        visualizations['Categorical Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 7. Tenure-based Churn Trend
    if 'tenure_months' in df.columns and 'churn_status' in df.columns:
        mapped = df['churn_status'].replace({"Yes": 1, "No": 0}).pipe(pd.to_numeric, errors='coerce')
        if not mapped.isnull().all():
            temp = pd.DataFrame({'tenure_months': df['tenure_months'], 'churn_status': mapped}).dropna()
            churn_by = temp.groupby('tenure_months')['churn_status'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=churn_by, x='tenure_months', y='churn_status', ax=ax, color='royalblue', marker='o')
            plt.title('Churn Rate Trend over Tenure (Months)')
            plt.ylabel('Average Churn Rate')
            plt.xlabel('Tenure (months)')
            plt.tight_layout()
            visualizations['Time-Series Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    return visualizations, round(churn_rate, 2), round(retention_rate, 2)
