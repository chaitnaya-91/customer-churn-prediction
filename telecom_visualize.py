import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
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
    # Load dataset
    df = pd.read_csv(csv_file_stream)
    df.columns = df.columns.str.lower()
    
    # Preprocess slightly to ensure clean plotting
    if 'monthlycharges' in df.columns:
        df['monthlycharges'] = pd.to_numeric(df['monthlycharges'], errors='coerce')
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

    # Calculate Churn & Retention Rate
    churn_rate = 0.0
    retention_rate = 0.0
    if 'churn' in df.columns:
        churn_mapped = df['churn'].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0})
        churn_mapped = pd.to_numeric(churn_mapped, errors='coerce').fillna(0).astype(int)
        counts = churn_mapped.value_counts()
        total = len(churn_mapped)
        if total > 0:
            churn_rate = float((counts.get(1, 0) / total) * 100)
            retention_rate = float((counts.get(0, 0) / total) * 100)

    visualizations = {}

    # 1. Descriptive Analysis
    # We can return HTML for this, not an image
    desc_html = df.describe().to_html(classes="min-w-full text-sm text-left text-gray-700", border=0)
    visualizations['Descriptive Analysis'] = {'type': 'html', 'content': desc_html}

    # 2. Distribution Analysis
    num_cols = ['tenure', 'monthlycharges', 'totalcharges']
    available_num_cols = [c for c in num_cols if c in df.columns]
    if available_num_cols:
        fig, axes = plt.subplots(1, len(available_num_cols), figsize=(15, 5))
        if len(available_num_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(available_num_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='skyblue')
            axes[idx].set_title(f'Distribution of {col}')
        plt.tight_layout()
        visualizations['Distribution Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 3. Correlation Analysis
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty and len(num_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        visualizations['Correlation Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 4. Missing Value Analysis
    missing = df.isnull().sum()
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing = missing[missing > 0]
        sns.barplot(x=missing.index, y=missing.values, ax=ax, palette='Reds_r')
        plt.title('Missing Values per Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['Missing Value Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}
    else:
        visualizations['Missing Value Analysis'] = {'type': 'text', 'content': 'No missing values found in the dataset.'}

    # 5. Outlier Analysis
    if available_num_cols:
        fig, axes = plt.subplots(1, len(available_num_cols), figsize=(15, 5))
        if len(available_num_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(available_num_cols):
            sns.boxplot(y=df[col].dropna(), ax=axes[idx], color='salmon')
            axes[idx].set_title(f'Outliers in {col}')
        plt.tight_layout()
        visualizations['Outlier Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 6. Categorical Analysis
    cat_cols = ['churn', 'internetservice', 'autopay']
    available_cat_cols = [c for c in cat_cols if c in df.columns]
    if available_cat_cols:
        fig, axes = plt.subplots(1, len(available_cat_cols), figsize=(15, 5))
        if len(available_cat_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(available_cat_cols):
            sns.countplot(x=df[col].dropna(), ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'Count of {col}')
        plt.tight_layout()
        visualizations['Categorical Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}

    # 7. Time-Series Analysis
    if 'tenure' in df.columns and 'churn' in df.columns:
        # Convert churn to numeric for plotting if it's not
        churn_mapped = df['churn'].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0})
        churn_mapped = pd.to_numeric(churn_mapped, errors='coerce')
        if not churn_mapped.isnull().all():
            temp_df = pd.DataFrame({'tenure': df['tenure'], 'churn': churn_mapped}).dropna()
            churn_by_tenure = temp_df.groupby('tenure')['churn'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=churn_by_tenure, x='tenure', y='churn', ax=ax, color='purple', marker='o')
            plt.title('Churn Rate Trend over Tenure (Time-Series Approximation)')
            plt.ylabel('Average Churn Rate')
            plt.xlabel('Tenure (months)')
            plt.tight_layout()
            visualizations['Time-Series Analysis'] = {'type': 'image', 'content': _fig_to_base64(fig)}
        else:
            visualizations['Time-Series Analysis'] = {'type': 'text', 'content': 'Could not process churn values for time-series analysis.'}
    else:
        visualizations['Time-Series Analysis'] = {'type': 'text', 'content': 'Required columns (tenure, churn) for Time-Series Analysis are missing.'}

    return visualizations, round(churn_rate, 2), round(retention_rate, 2)
