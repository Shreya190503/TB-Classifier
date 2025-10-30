#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TB Diagnosis AutoML - Streamlit App (Saves ALL Models)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Page configuration
st.set_page_config(
    page_title="TB Diagnosis AutoML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .best-model {
        border-left: 5px solid #1f77b4;
        background-color: #fff5f5;
        color: #000000 !important;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAutoML:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def create_sample_data(self):
        """Create realistic TB gene expression data"""
        np.random.seed(42)
        
        # 200 patients: 100 Active TB, 100 Latent TB
        n_samples = 200
        n_genes = 300
        
        # Gene expression data
        X = np.random.randn(n_samples, n_genes)
        
        # Biological patterns for Active TB
        X[:100, :30] += 2.0    # Upregulated in Active TB
        X[:100, 150:180] -= 1.5 # Downregulated in Active TB
        X[:100, 250:280] += 1.2 # Biomarker genes
        
        # Latent TB patterns
        X[100:, :30] -= 1.0
        X[100:, 150:180] += 1.0
        
        # Target: Active TB = 1, Latent TB = 0
        y = np.array([1] * 100 + [0] * 100)
        
        # Gene names
        gene_names = [f'GENE_{i:04d}' for i in range(1, n_genes + 1)]
        
        return X, y, gene_names
    
    def initialize_models(self):
        """Initialize 8 different ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Gaussian Naive Bayes': GaussianNB()
        }
        return self.models
    
    def train_all_models(self, X, y):
        """Train all models and save each one"""
        st.info("🔄 Training all models... This may take a few moments.")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.feature_names = [f'GENE_{i:04d}' for i in range(1, X.shape[1] + 1)]
        
        best_accuracy = 0
        best_model_name = None
        
        # Train each model
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            # Save individual model
            self.save_model(name, model, accuracy)
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
            
            # Update progress
            progress_bar.progress((i + 1) / len(self.models))
            time.sleep(0.5)  # Visual effect
        
        status_text.text("✅ All models trained successfully!")
        
        # Save scaler and metadata
        self.save_scaler_and_metadata(best_model_name, best_accuracy)
        
        return best_model_name, best_accuracy
    
    def save_model(self, model_name, model, accuracy):
        """Save individual model to file"""
        filename = f"{self.models_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
        model_data = {
            'model': model,
            'model_name': model_name,
            'accuracy': accuracy,
            'feature_names': self.feature_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def save_scaler_and_metadata(self, best_model_name, best_accuracy):
        """Save scaler and overall metadata"""
        metadata = {
            'scaler': self.scaler,
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy,
            'feature_names': self.feature_names,
            'all_results': self.results,
            'models_trained': list(self.models.keys())
        }
        
        with open(f"{self.models_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_saved_models(self):
        """Load all saved models"""
        loaded_models = {}
        metadata_path = f"{self.models_dir}/metadata.pkl"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Load individual models
            for model_name in metadata['models_trained']:
                filename = f"{self.models_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        loaded_models[model_name] = pickle.load(f)
            
            return loaded_models, metadata
        return None, None
    
    def predict_with_model(self, model_name, features):
        """Make prediction using a specific saved model"""
        filename = f"{self.models_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
        metadata_path = f"{self.models_dir}/metadata.pkl"
        
        if os.path.exists(filename) and os.path.exists(metadata_path):
            # Load model
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load metadata for scaler
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Prepare features
            features_df = pd.DataFrame([features], columns=metadata['feature_names'])
            features_scaled = metadata['scaler'].transform(features_df)
            
            # Predict
            prediction = model_data['model'].predict(features_scaled)[0]
            probability = model_data['model'].predict_proba(features_scaled)[0]
            
            diagnosis = "Active Tuberculosis" if prediction == 1 else "Latent Tuberculosis"
            confidence = probability[1] if prediction == 1 else probability[0]
            
            return {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probabilities': {
                    'active_tb': probability[1],
                    'latent_tb': probability[0]
                },
                'model_used': model_name,
                'model_accuracy': model_data['accuracy']
            }
        
        return None

# Initialize AutoML
automl = StreamlitAutoML()

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 TB Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning Powered Tuberculosis Diagnosis")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🏠 Home", "🤖 Train Models", "🔮 Make Predictions", "📊 Model Comparison"]
    )
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "🤖 Train Models":
        train_models()
    elif app_mode == "🔮 Make Predictions":
        make_predictions()
    elif app_mode == "📊 Model Comparison":
        model_comparison()

def show_home():
    st.markdown("""
    ## Welcome to the TB Diagnosis AutoML System!
    
    This application uses machine learning to diagnose tuberculosis based on gene expression data.
    
    ### Features:
    - **Train 8 different ML models** simultaneously
    - **Save all trained models** for future use
    - **Compare model performance** with visualizations
    - **Make predictions** using any saved model
    - **Export results** and models
    
    ### How to use:
    1. **Train Models**: Train all ML models on TB gene expression data
    2. **Make Predictions**: Use trained models to diagnose new patients
    3. **Compare Models**: See which model performs best
    
    ### Available Models:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    models_list = [
        "Random Forest", "Logistic Regression", "Support Vector Machine", "Gradient Boosting",
        "Decision Tree", "K-Nearest Neighbors", "AdaBoost", "Gaussian Naive Bayes"
    ]
    
    for i, model in enumerate(models_list):
        with [col1, col2, col3, col4][i % 4]:
            st.info(f"• {model}")
    
    # Check if models already exist
    if os.path.exists(f"{automl.models_dir}/metadata.pkl"):
        st.success("✅ Models are already trained and saved!")
        st.info("Go to 'Make Predictions' to diagnose patients or 'Model Comparison' to see performance.")

def train_models():
    st.header("🤖 Train Machine Learning Models")
    
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Creating sample TB dataset..."):
            X, y, feature_names = automl.create_sample_data()
        
        st.success(f"✅ Dataset created: {X.shape[0]} patients, {X.shape[1]} genes")
        
        # Initialize and train models
        automl.initialize_models()
        best_model, best_accuracy = automl.train_all_models(X, y)
        
        # Display results
        st.success(f"🎉 Training Complete! Best Model: **{best_model}** (Accuracy: {best_accuracy:.3f})")
        
        # Show quick results
        st.subheader("Quick Results")
        results_data = []
        for name, result in automl.results.items():
            results_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.3f}",
                'CV Score': f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(results_data))
        
        # Saved models info
        st.subheader("💾 Saved Models")
        model_files = [f for f in os.listdir(automl.models_dir) if f.endswith('.pkl')]
        for model_file in model_files:
            st.write(f"• `{model_file}`")

def make_predictions():
    st.header("🔮 Make Predictions")
    
    # Check if models exist
    if not os.path.exists(f"{automl.models_dir}/metadata.pkl"):
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    # Load available models
    loaded_models, metadata = automl.load_saved_models()
    
    if loaded_models:
        st.success(f"✅ Loaded {len(loaded_models)} trained models")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Model for Prediction",
            list(loaded_models.keys())
        )
        
        # Display model info
        model_info = loaded_models[selected_model]
        st.info(f"**{selected_model}** - Accuracy: {model_info['accuracy']:.3f}")
        
        # Input methods
        input_method = st.radio("Input Method", ["Manual Input", "CSV Upload"])
        
        if input_method == "Manual Input":
            manual_prediction(selected_model, metadata)
        else:
            csv_prediction(selected_model, metadata)

def manual_prediction(model_name, metadata):
    st.subheader("Manual Patient Data Input")
    
    # Create sample data or manual input
    if st.button("🎲 Generate Sample Patient Data"):
        sample_features = np.random.randn(len(metadata['feature_names']))
        sample_features[:30] += 2.0  # Active TB pattern
        
        # Display first 10 features for manual adjustment
        st.info("Sample patient data generated (Active TB pattern)")
    
    # Feature input (show first 10 as example)
    st.write("Adjust gene expression values (first 10 features shown):")
    
    features = []
    cols = st.columns(2)
    
    for i in range(min(10, len(metadata['feature_names']))):
        with cols[i % 2]:
            default_val = np.random.normal(0, 1)
            if i < 5:  # Make first few look like Active TB
                default_val += 2.0
            feature_val = st.number_input(
                f"{metadata['feature_names'][i]}",
                value=float(default_val),
                step=0.1
            )
            features.append(feature_val)
    
    # Fill remaining features with random values
    remaining_features = np.random.normal(0, 1, len(metadata['feature_names']) - 10)
    features.extend(remaining_features)
    
    if st.button("🩺 Diagnose Patient", type="primary"):
        with st.spinner("Analyzing patient data..."):
            result = automl.predict_with_model(model_name, features)
        
        if result:
            display_prediction_result(result)

def csv_prediction(model_name, metadata):
    st.subheader("CSV File Upload")
    
    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {len(df)} patients, {len(df.columns)} columns")
            st.write("First few rows of your data:")
            st.dataframe(df.head())
            
            # Show data types to help debug
            st.write("Data types in your CSV:")
            st.write(df.dtypes)
            
            # Identify numeric columns automatically
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ No numeric columns found in the CSV file!")
                st.info("Please make sure your CSV contains numerical gene expression data.")
                return
            
            st.info(f"🔍 Found {len(numeric_columns)} numeric columns for prediction")
            
            if st.button("🔮 Diagnose All Patients", type="primary"):
                results = []
                
                for idx, row in df.iterrows():
                    try:
                        # Extract only numeric values from the row
                        features = []
                        for col in numeric_columns:
                            try:
                                # Convert to float, handle missing values
                                value = float(row[col]) if pd.notna(row[col]) else 0.0
                                features.append(value)
                            except (ValueError, TypeError):
                                features.append(0.0)  # Default value for conversion errors
                        
                        # Ensure we have the right number of features
                        if len(features) < len(metadata['feature_names']):
                            # Pad with zeros if needed
                            features.extend([0.0] * (len(metadata['feature_names']) - len(features)))
                        elif len(features) > len(metadata['feature_names']):
                            # Trim if too many features
                            features = features[:len(metadata['feature_names'])]
                        
                        # Make prediction
                        result = automl.predict_with_model(model_name, features)
                        if result:
                            result['patient_id'] = f"Patient_{idx+1}"
                            # Include original sample ID if available
                            if 'sample_id' in df.columns or 'Sample_ID' in df.columns:
                                sample_col = 'sample_id' if 'sample_id' in df.columns else 'Sample_ID'
                                result['original_id'] = str(row[sample_col])
                            results.append(result)
                            
                    except Exception as e:
                        st.warning(f"⚠️ Error processing patient {idx+1}: {str(e)}")
                        continue
                
                # Display all results
                if results:
                    st.success(f"✅ Successfully processed {len(results)} patients")
                    for result in results:
                        display_prediction_result(result)
                else:
                    st.error("❌ No successful predictions. Please check your CSV format.")
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("""
            **CSV File Requirements:**
            - Must contain numerical data (gene expression values)
            - Can have sample IDs in separate columns
            - First row can be headers
            - Missing values will be filled with zeros
            """)
def display_prediction_result(result):
    """Display prediction results in a nice format"""
    is_active = "Active" in result['diagnosis']
    
    if is_active:
        st.error(f"""
        🔴 **Diagnosis: {result['diagnosis']}**
        - Confidence: {result['confidence']:.3f}
        - Model: {result['model_used']}
        - Model Accuracy: {result['model_accuracy']:.3f}
        """)
    else:
        st.success(f"""
        🟢 **Diagnosis: {result['diagnosis']}**
        - Confidence: {result['confidence']:.3f}
        - Model: {result['model_used']}
        - Model Accuracy: {result['model_accuracy']:.3f}
        """)
    
    # Probability chart
    prob_data = pd.DataFrame({
        'Condition': ['Active TB', 'Latent TB'],
        'Probability': [result['probabilities']['active_tb'], result['probabilities']['latent_tb']]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(prob_data['Condition'], prob_data['Probability'], color=['#ff6b6b', '#4ecdc4'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Diagnosis Probabilities')
    
    # Add value labels
    for bar, value in zip(bars, prob_data['Probability']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center')
    
    st.pyplot(fig)

def model_comparison():
    st.header("📊 Model Comparison")
    
    if not os.path.exists(f"{automl.models_dir}/metadata.pkl"):
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    with open(f"{automl.models_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    st.success(f"✅ Loaded results for {len(metadata['all_results'])} models")
    
    # Comparison chart
    comparison_data = []
    for name, result in metadata['all_results'].items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'CV_Score': result['cv_mean'],
            'CV_Std': result['cv_std']  # Add standard deviation
        })
    
    df_comparison = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    # Display comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.dataframe(df_comparison.style.format({
            'Accuracy': '{:.3f}',
            'CV_Score': '{:.3f}',
            'CV_Std': '{:.3f}'
        }).highlight_max(axis=0, color='lightgreen'))
    
    with col2:
        st.subheader("Best Model")
        best_model = metadata['best_model_name']
        best_accuracy = metadata['best_accuracy']
        
        st.markdown(f"""
        <div class='model-card best-model'>
            <h3 style='color: #000000;'>🏆 {best_model}</h3>
            <p><strong>Accuracy:</strong> {best_accuracy:.3f}</p>
            <p><strong>CV Score:</strong> {metadata['all_results'][best_model]['cv_mean']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization - FIXED VERSION
    st.subheader("Performance Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison - FIXED
    df_sorted = df_comparison.sort_values('Accuracy')
    bars1 = ax1.barh(df_sorted['Model'], df_sorted['Accuracy'], color='skyblue', alpha=0.7)
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', ha='left')
    
    # CV score comparison - FIXED (no error bars for simplicity)
    df_sorted_cv = df_comparison.sort_values('CV_Score')
    bars2 = ax2.barh(df_sorted_cv['Model'], df_sorted_cv['CV_Score'], 
                     color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Cross-Validation Score')
    ax2.set_title('CV Scores (Mean)')
    ax2.set_xlim(0, 1)
    
    # Add value labels on bars
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', ha='left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional visualization with error bars (optional)
    st.subheader("CV Scores with Error Bars")
    
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    # Create proper error bars using plt.barh with yerr parameter
    y_pos = np.arange(len(df_comparison))
    
    # FIXED: Use proper error bar format
    ax3.barh(y_pos, df_comparison['CV_Score'], 
             xerr=df_comparison['CV_Std'],  # This should work now
             capsize=5, alpha=0.7, color='lightgreen')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(df_comparison['Model'])
    ax3.set_xlabel('Cross-Validation Score')
    ax3.set_title('CV Scores with Standard Deviation')
    ax3.set_xlim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Download models
    st.subheader("💾 Download Models")
    st.write("All models are saved in the `saved_models/` directory:")
    
    model_files = [f for f in os.listdir(automl.models_dir) if f.endswith('.pkl')]
    for model_file in model_files:
        file_path = os.path.join(automl.models_dir, model_file)
        with open(file_path, 'rb') as f:
            st.download_button(
                label=f"Download {model_file}",
                data=f,
                file_name=model_file,
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()