import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Solar Panel Hotspot Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #ff7b00 0%, #ff9500 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .categorical-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Move the cached function outside the class
@st.cache_resource
def load_model():
    """Load the Ridge regression model"""
    try:
        # Try to load from different possible locations
        possible_paths = [
            'ridge_regressor_model.pkl',
            'ridge_regressor_model.joblib',
            'model/ridge_regressor_model.pkl',
            'model/ridge_regressor_model.joblib',
            './ridge_regressor_model.pkl',
            './ridge_regressor_model.joblib'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Model file not found. Please upload ridge_regressor_model.pkl or ridge_regressor_model.joblib")
        
        # Load model based on file extension
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        
        # Try to load feature names
        feature_names = None
        feature_paths = ['feature_names.pkl', 'model/feature_names.pkl', './feature_names.pkl']
        for path in feature_paths:
            if os.path.exists(path):
                with open(path, 'rb') as file:
                    feature_names = pickle.load(file)
                break
        
        return model, feature_names, True
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

class HotspotRatioPredictor:
    def __init__(self):
        # Define feature information
        self.numerical_features = {
            'Voc': {'min': 0.0, 'max': 50.0, 'unit': 'V', 'default': 36.5, 'step': 0.1},
            'Isc': {'min': 0.0, 'max': 15.0, 'unit': 'A', 'default': 9.2, 'step': 0.1},
            'Pmax': {'min': 0.0, 'max': 500.0, 'unit': 'W', 'default': 300.0, 'step': 1.0},
            'FF': {'min': 0.0, 'max': 1.0, 'unit': 'ratio', 'default': 0.75, 'step': 0.01},
            'Efficiency': {'min': 0.0, 'max': 25.0, 'unit': '%', 'default': 18.5, 'step': 0.1},
            'Field Age (Year)': {'min': 0, 'max': 30, 'unit': 'years', 'default': 5, 'step': 1}
        }
        
        self.categorical_features = [
            'EVA yellowing',
            'Delamination', 
            'Junction box breakage',
            'Dried algae',
            'Dust and Soiling',
            'Bird droppings'
        ]
        
        # Load model using the cached function
        self.model, self.feature_names, self.model_loaded = load_model()
    
    def predict_hotspot_ratio(self, values):
        """Make prediction"""
        if not self.model_loaded:
            return None, "Model not loaded"
        
        try:
            # Create DataFrame with the input values
            input_data = pd.DataFrame([values])
            
            # Ensure columns are in the correct order
            if self.feature_names:
                # Use the saved feature names order
                input_data = input_data[self.feature_names]
            else:
                # Use expected order if feature names not available
                expected_order = list(self.numerical_features.keys()) + self.categorical_features
                input_data = input_data[expected_order]
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            return prediction, None
            
        except Exception as e:
            return None, str(e)
    
    def get_hotspot_classification(self, hotspot_ratio):
        """Classify hotspot severity"""
        if hotspot_ratio < 0.1:
            return "Very Low Risk", "#4CAF50"
        elif hotspot_ratio < 0.3:
            return "Low Risk", "#8BC34A"
        elif hotspot_ratio < 0.5:
            return "Moderate Risk", "#FFC107"
        elif hotspot_ratio < 0.7:
            return "High Risk", "#FF9800"
        else:
            return "Critical Risk", "#F44336"
    
    def create_gauge_chart(self, value):
        """Create a gauge chart for the prediction"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hotspot Ratio"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1.0]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 0.1], 'color': "#E8F5E8"},
                    {'range': [0.1, 0.3], 'color': "#C8E6C9"},
                    {'range': [0.3, 0.5], 'color': "#FFF9C4"},
                    {'range': [0.5, 0.7], 'color': "#FFCC02"},
                    {'range': [0.7, 1.0], 'color': "#FFCDD2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'input_values' not in st.session_state:
            st.session_state.input_values = {}
            
            # Initialize numerical features
            for feature, info in self.numerical_features.items():
                st.session_state.input_values[feature] = info['default']
            
            # Initialize categorical features (default to 0)
            for feature in self.categorical_features:
                st.session_state.input_values[feature] = 0
    
    def run_app(self):
        """Run the Streamlit app"""
        
        # Initialize session state
        self.initialize_session_state()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☀️ Solar Panel Hotspot Ratio Predictor</h1>
            <p>Predict hotspot formation risk using machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please ensure 'ridge_regressor_model.pkl' or 'ridge_regressor_model.joblib' is in the app directory.")
            st.stop()
        else:
            st.success("✅ Ridge Regression Model loaded successfully!")
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("⚡ Solar Panel Parameters")
            
            # Numerical parameters
            input_values = {}
            
            for feature, info in self.numerical_features.items():
                current_value = st.session_state.input_values.get(feature, info['default'])
                
                input_values[feature] = st.number_input(
                    f"{feature} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(current_value),
                    step=float(info['step']),
                    key=f"input_{feature}",
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="categorical-section">', unsafe_allow_html=True)
            st.subheader("🔍 Defect Indicators")
            st.write("Select if the following defects are present (1) or absent (0):")
            
            # Categorical parameters
            for feature in self.categorical_features:
                current_value = st.session_state.input_values.get(feature, 0)
                
                input_values[feature] = st.selectbox(
                    f"{feature}",
                    options=[0, 1],
                    index=current_value,
                    key=f"input_{feature}",
                    help="0 = Absent, 1 = Present"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state with current input values
        st.session_state.input_values = input_values
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔮 Predict Hotspot Ratio", type="primary", use_container_width=True):
                prediction, error = self.predict_hotspot_ratio(input_values)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state['prediction'] = prediction
                    st.session_state['prediction_input_values'] = input_values
        
        # Display results
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            
            # Ensure prediction is within valid range
            prediction = max(0, min(1, prediction))
            
            # Results card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prediction Results</h2>
                <h1 style="font-size: 3rem; margin: 0.5rem 0;">{prediction:.4f}</h1>
                <p style="font-size: 1.2rem; opacity: 0.9;">
                    Hotspot Ratio (0 = No hotspot, 1 = Maximum hotspot)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart and classification
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = self.create_gauge_chart(prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                classification, color = self.get_hotspot_classification(prediction)
                st.markdown(f"""
                <div style="background: {color}; padding: 2rem; border-radius: 10px; 
                            text-align: center; color: white; margin-top: 2rem;">
                    <h3>⚠️ Risk Level</h3>
                    <h2>{classification}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Input summary
            st.subheader("📋 Input Parameters Summary")
            
            # Create separate DataFrames for numerical and categorical features
            numerical_data = []
            categorical_data = []
            
            for feature, value in st.session_state.get('prediction_input_values', {}).items():
                if feature in self.numerical_features:
                    unit = self.numerical_features[feature]['unit']
                    numerical_data.append({
                        'Parameter': feature,
                        'Value': f"{value:.2f}",
                        'Unit': unit
                    })
                else:
                    status = "Present" if value == 1 else "Absent"
                    categorical_data.append({
                        'Defect': feature,
                        'Status': status,
                        'Value': value
                    })
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if numerical_data:
                    st.write("**Numerical Parameters:**")
                    numerical_df = pd.DataFrame(numerical_data)
                    st.dataframe(numerical_df, use_container_width=True)
            
            with col2:
                if categorical_data:
                    st.write("**Defect Indicators:**")
                    categorical_df = pd.DataFrame(categorical_data)
                    st.dataframe(categorical_df, use_container_width=True)
            
            # Recommendations based on prediction
            st.subheader("💡 Recommendations")
            
            if prediction < 0.1:
                st.success("✅ **Excellent condition!** The solar panel shows minimal risk of hotspot formation. Continue regular maintenance.")
            elif prediction < 0.3:
                st.info("ℹ️ **Good condition.** Monitor the panel regularly and address any emerging issues promptly.")
            elif prediction < 0.5:
                st.warning("⚠️ **Moderate risk detected.** Consider increased monitoring and preventive maintenance.")
            elif prediction < 0.7:
                st.warning("🔶 **High risk!** Immediate inspection and maintenance recommended to prevent hotspot formation.")
            else:
                st.error("🚨 **Critical risk!** Urgent action required. The panel may already have or be developing hotspots.")
            
            # Additional info
            st.info("💡 **Note:** This prediction is based on a Ridge regression model trained on solar panel performance and defect data. Results should be verified through thermal imaging and professional inspection.")

# Run the app
if __name__ == "__main__":
    app = HotspotRatioPredictor()
    app.run_app()