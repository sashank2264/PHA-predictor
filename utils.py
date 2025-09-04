import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# Dictionary of wastewater types, organic materials, and their corresponding value-added products
wastewater_info = {
    "Municipal Wastewater": {
        "organic_materials": {
            "Carbohydrates": ["Biogas (Methane)", "Biodiesel", "Bioethanol", "PHA production"],
            "Proteins": ["Biogas (Methane)", "Biodegradable Plastics", "PHA production", "Animal Feed"],
            "Fats, Oils, and Grease (FOG)": ["Biodiesel", "Biogas (Methane)", "PHA production"],
            "Amino Acids and Peptides": ["Biogas (Methane)", "Animal Feed", "PHA production"]
        }
    },
    "Industrial Wastewater": {
        "organic_materials": {
            "Phenols and Aromatic Compounds": ["Biogas (Methane)", "Biodiesel", "Fine Chemicals (e.g., phenols for plastic production)", "PHA production"],
            "Carbohydrates": ["Biogas (Methane)", "Bioethanol", "PHA production"],
            "Proteins": ["Biogas (Methane)", "Biodegradable Plastics", "PHA production"],
            "Fats, Oils, and Grease (FOG)": ["Biodiesel", "Biogas (Methane)", "PHA production"]
        }
    },
    "Agricultural Wastewater": {
        "organic_materials": {
            "Carbohydrates": ["Biogas (Methane)", "Bioethanol", "PHA production"],
            "Cellulose": ["Biochar", "PHA production", "Biogas (Methane)"],
            "Proteins": ["Biogas (Methane)", "Bioethanol", "PHA production"]
        }
    },
    "University (College) Wastewater": {
        "organic_materials": {
            "Carbohydrates": ["Biogas (Methane)", "Biodiesel", "Bioethanol", "PHA production"],
            "Proteins": ["Biogas (Methane)", "Biodegradable Plastics", "PHA production", "Animal Feed"],
            "Fats, Oils, and Grease (FOG)": ["Biodiesel", "Biogas (Methane)", "PHA production"],
            "Amino Acids": ["Biogas (Methane)", "Animal Feed", "PHA production"]
        }
    },
    "Stormwater Runoff": {
        "organic_materials": {
            "Carbohydrates": ["Biogas (Methane)", "Bioethanol", "PHA production"],
            "Proteins": ["Biogas (Methane)", "PHA production"],
            "Fats, Oils, and Grease (FOG)": ["Biodiesel", "Biogas (Methane)", "PHA production"]
        }
    }
}

def get_value_added_products(wastewater_type, organic_material):
    """
    Get value-added products for a given wastewater type and organic material.
    
    Args:
        wastewater_type (str): Type of wastewater
        organic_material (str): Type of organic material
        
    Returns:
        list: List of value-added products
    """
    if wastewater_type in wastewater_info and organic_material in wastewater_info[wastewater_type]["organic_materials"]:
        return wastewater_info[wastewater_type]["organic_materials"][organic_material]
    return []

def plot_correlation_matrix(data):
    """
    Create a correlation matrix visualization.
    
    Args:
        data (DataFrame): Input data
        
    Returns:
        plotly.Figure: Correlation matrix plot
    """
    # Filter numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='Viridis',
        title='Correlation Between Features'
    )
    
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Create an actual vs predicted plot.
    
    Args:
        y_test (Series): Actual values
        y_pred (array): Predicted values
        
    Returns:
        matplotlib.Figure: Actual vs predicted plot
    """
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    
    # Add reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('Actual PHA Yield (g/L)')
    ax.set_ylabel('Predicted PHA Yield (g/L)')
    ax.set_title('Random Forest: Actual vs Predicted PHA Yield')
    ax.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax.annotate(f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df):
    """
    Create a feature importance plot.
    
    Args:
        importance_df (DataFrame): Feature importance data
        
    Returns:
        matplotlib.Figure: Feature importance plot
    """
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot using seaborn
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df, 
        palette="viridis", 
        ax=ax
    )
    
    # Add labels and title
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in Random Forest Model")
    
    # Add grid lines
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_distribution(data, column, hue=None):
    """
    Create a distribution plot.
    
    Args:
        data (DataFrame): Input data
        column (str): Column to plot
        hue (str, optional): Column for color encoding
        
    Returns:
        matplotlib.Figure: Distribution plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hue is not None and hue in data.columns:
        sns.histplot(data=data, x=column, hue=hue, kde=True, ax=ax)
    else:
        sns.histplot(data=data, x=column, kde=True, ax=ax)
    
    ax.set_title(f"Distribution of {column}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
