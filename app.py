import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO
import os
from models import (
    preprocess_data, 
    train_models, 
    predict_pha_yield_and_type,
    get_model_performance,
    get_feature_importance
)
from utils import (
    get_value_added_products,
    wastewater_info,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_distribution
)

# Page configuration
st.set_page_config(
    page_title="PHA Production Predictor",
    page_icon="🧪",
    layout="wide"
)

# Title and introduction
st.title("🧪 PHA Production from Wastewater Analysis")
st.markdown("""
This application helps you predict Polyhydroxyalkanoates (PHA) production from wastewater 
using machine learning models. Upload your data or use the sample dataset to:
- Predict PHA yield and type
- Analyze feature importance
- Visualize data relationships
- Get value-added product recommendations
""")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'yield_model' not in st.session_state:
    st.session_state.yield_model = None
if 'type_model' not in st.session_state:
    st.session_state.type_model = None
if 'features' not in st.session_state:
    st.session_state.features = ['Carbon Source Quantity (g/L)', 'Time (h)', 'Temperature (°C)', 'pH', 'C/N Ratio', 'Nitrogen Source Quantity (g/L)']
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Sidebar for data upload and model options
with st.sidebar:
    st.header("Data & Models")
    
    upload_option = st.radio(
        "Choose data source:",
        ["Upload your data", "Use sample data"]
    )
    
    if upload_option == "Upload your data":
        uploaded_file = st.file_uploader("Upload your PHA production dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:  # Excel file
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"Data loaded successfully: {data.shape[0]} rows and {data.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        # Use sample data
        try:
            # Load sample data
            sample_data = pd.read_csv("sample_data.csv")
            st.session_state.data = sample_data
            st.success(f"Sample data loaded: {sample_data.shape[0]} rows and {sample_data.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    # Model training button
    if st.session_state.data is not None:
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a moment."):
                # Preprocess data
                X_train, X_test, y_train, y_test, y_type_train, y_type_test = preprocess_data(st.session_state.data)
                
                # Train models
                yield_model, type_model = train_models(X_train, y_train, y_type_train)
                
                # Save models to session state
                st.session_state.yield_model = yield_model
                st.session_state.type_model = type_model
                
                # Calculate and save model performance metrics
                st.session_state.model_metrics = get_model_performance(
                    yield_model, type_model, X_test, y_test, y_type_test
                )
                
                st.success("Models trained successfully!")

# Main content area
if st.session_state.data is not None:
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", 
        "Predict PHA Production", 
        "Model Performance", 
        "Feature Analysis",
        "Value-Added Products"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Data Overview")
        
        # Display data info
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head())
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        st.dataframe(st.session_state.data.describe())
        
        # Display correlation heatmap
        st.subheader("Correlation Matrix")
        fig = plot_correlation_matrix(st.session_state.data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data distribution plots
        st.subheader("Data Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            selected_column = st.selectbox("Select column for histogram:", numeric_columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.data[selected_column], kde=True, ax=ax)
            plt.title(f"Distribution of {selected_column}")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            if 'PHA Type' in st.session_state.data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                type_counts = st.session_state.data['PHA Type'].value_counts()
                sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax)
                plt.title("Distribution of PHA Types")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    # Tab 2: Predict PHA Production
    with tab2:
        st.header("Predict PHA Production")
        
        if st.session_state.yield_model is None or st.session_state.type_model is None:
            st.warning("Please train the models first using the button in the sidebar.")
        else:
            st.subheader("Enter parameters for PHA prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                carbon = st.number_input("Carbon Source Quantity (g/L)", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
                time = st.number_input("Time (h)", min_value=0.0, max_value=1000.0, value=48.0, step=1.0)
                temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            
            with col2:
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                nitrogen = st.number_input("Nitrogen Source Quantity (g/L)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
                cn_ratio = carbon / nitrogen if nitrogen > 0 else 0
                st.metric("Calculated C/N Ratio", f"{cn_ratio:.2f}")
            
            # Make prediction on button click
            if st.button("Predict"):
                with st.spinner("Generating prediction..."):
                    # Create input DataFrame
                    input_data = pd.DataFrame({
                        'Carbon Source Quantity (g/L)': [carbon],
                        'Time (h)': [time],
                        'Temperature (°C)': [temp],
                        'pH': [ph],
                        'C/N Ratio': [cn_ratio],
                        'Nitrogen Source Quantity (g/L)': [nitrogen]
                    })
                    
                    # Get predictions
                    predicted_yield, predicted_type = predict_pha_yield_and_type(
                        st.session_state.yield_model, 
                        st.session_state.type_model, 
                        input_data
                    )
                    
                    # Display predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted PHA Yield", 
                            f"{predicted_yield[0]:.3f} g/L",
                            help="Predicted production of PHA in grams per liter"
                        )
                    with col2:
                        st.metric(
                            "Predicted PHA Type", 
                            predicted_type[0],
                            help="Predicted type of PHA polymer"
                        )
                        
                    st.info("These predictions are based on the trained Random Forest models.")
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance")
        
        if st.session_state.model_metrics is None:
            st.warning("Please train the models first using the button in the sidebar.")
        else:
            # Display model performance metrics
            metrics = st.session_state.model_metrics
            
            # Display regression metrics
            st.subheader("PHA Yield Prediction (Regression) Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.4f}")
                st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
                st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}")
            
            # Display actual vs predicted plot
            st.subheader("Actual vs Predicted PHA Yield")
            fig = plot_actual_vs_predicted(metrics['y_test'], metrics['y_pred'])
            st.pyplot(fig)
            
            # Display classification metrics
            st.subheader("PHA Type Prediction (Classification) Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                st.metric("Precision (Macro Avg)", f"{metrics['precision']:.4f}")
            with col2:
                st.metric("Recall (Macro Avg)", f"{metrics['recall']:.4f}")
                st.metric("F1 Score (Macro Avg)", f"{metrics['f1']:.4f}")
            
            # Display classification report
            st.subheader("Detailed Classification Report")
            st.code(metrics['classification_report'])
    
    # Tab 4: Feature Analysis
    with tab4:
        st.header("Feature Importance Analysis")
        
        if st.session_state.yield_model is None:
            st.warning("Please train the models first using the button in the sidebar.")
        else:
            st.subheader("Feature Importance in PHA Yield Prediction")
            
            # Get and display feature importance
            importances = get_feature_importance(st.session_state.yield_model, st.session_state.features)
            fig = plot_feature_importance(importances)
            st.pyplot(fig)
            
            # Explanation
            st.markdown("""
            ### Understanding Feature Importance
            
            The chart above displays the relative importance of each feature in predicting PHA yield.
            Features with higher importance scores have more influence on the prediction results.
            
            - Higher bars indicate more important features
            - The model primarily relies on these features to make accurate predictions
            - Feature importance helps in understanding which parameters should be controlled more precisely in the production process
            """)
            
            # Feature interaction effects
            st.subheader("Feature Interaction Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature:", st.session_state.features)
            with col2:
                y_feature = st.selectbox("Y-axis feature:", st.session_state.features, index=1)
            
            # Create scatter plot with PHA Production as color
            if 'PHA Production (g/L)' in st.session_state.data.columns:
                fig = px.scatter(
                    st.session_state.data, 
                    x=x_feature, 
                    y=y_feature, 
                    color='PHA Production (g/L)',
                    title=f"Relationship between {x_feature} and {y_feature}",
                    color_continuous_scale="Viridis",
                    opacity=0.7,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Value-Added Products
    with tab5:
        st.header("Value-Added Products from Wastewater")
        
        # Select wastewater type and organic material
        st.subheader("Select Wastewater Type and Organic Material")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wastewater_type = st.selectbox(
                "Select Wastewater Type:",
                list(wastewater_info.keys())
            )
        
        with col2:
            if wastewater_type:
                organic_materials = list(wastewater_info[wastewater_type]["organic_materials"].keys())
                organic_material = st.selectbox(
                    "Select Organic Material:",
                    organic_materials
                )
        
        # Display value-added products
        if wastewater_type and organic_material:
            st.subheader(f"Value-Added Products from {organic_material} in {wastewater_type}")
            
            products = get_value_added_products(wastewater_type, organic_material)
            
            # Create columns for products
            cols = st.columns(min(3, len(products)))
            for i, product in enumerate(products):
                with cols[i % 3]:
                    st.info(f"**{product}**")
                    
                    # Add specific information based on product type
                    if product == "PHA production":
                        st.write("A biodegradable biopolymer that can replace petroleum-based plastics.")
                    elif product == "Biogas (Methane)":
                        st.write("Can be used for energy generation or further upgraded to natural gas standards.")
                    elif product == "Biodiesel":
                        st.write("Renewable fuel suitable for diesel engines with lower carbon emissions.")
                    elif product == "Bioethanol":
                        st.write("Renewable fuel that can be blended with gasoline.")
            
            # Display recommendations
            st.subheader("Recommendations")
            st.write(f"""
            Based on your selection of {wastewater_type} and {organic_material}, the following value-added products can be produced.
            The PHA prediction model in this application can help optimize conditions for PHA production if you choose that pathway.
            """)
            
            # Example process flow diagram
            st.subheader("Simplified Process Flow for Product Recovery")
            
            process_steps = {
                "PHA production": [
                    "Wastewater Collection", 
                    "Bacterial Cultivation", 
                    "PHA Accumulation", 
                    "Cell Harvesting", 
                    "PHA Extraction", 
                    "Purification"
                ],
                "Biogas (Methane)": [
                    "Wastewater Collection", 
                    "Anaerobic Digestion", 
                    "Biogas Capture", 
                    "Purification", 
                    "Utilization"
                ],
                "Biodiesel": [
                    "Wastewater Collection", 
                    "Lipid Extraction", 
                    "Transesterification", 
                    "Separation", 
                    "Purification"
                ],
                "Bioethanol": [
                    "Wastewater Collection", 
                    "Pretreatment", 
                    "Fermentation", 
                    "Distillation", 
                    "Dehydration"
                ]
            }
            
            selected_product = st.selectbox("Select product to view process flow:", products)
            
            if selected_product in process_steps:
                steps = process_steps[selected_product]
                
                # Create a horizontal flow diagram
                cols = st.columns(len(steps))
                for i, step in enumerate(steps):
                    with cols[i]:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(f"<div style='text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>{step}</div>", unsafe_allow_html=True)
                        if i < len(steps) - 1:
                            st.markdown("<div style='text-align: center; margin-top: 20px;'>↓</div>", unsafe_allow_html=True)
else:
    # No data loaded yet
    st.info("Please upload a dataset or use the sample data to get started.")
    
    # Show placeholder image and info
    st.markdown("""
    ## About PHA Production from Wastewater
    
    Polyhydroxyalkanoates (PHAs) are biodegradable polymers produced by various bacteria as energy storage materials. 
    They can be produced from organic matter in wastewater, making them a sustainable alternative to petroleum-based plastics.
    
    This application helps you:
    
    1. Predict PHA yield and type based on process parameters
    2. Analyze important factors affecting PHA production
    3. Explore different value-added products from various wastewater types
    
    Upload your data or use our sample dataset to get started.
    """)
