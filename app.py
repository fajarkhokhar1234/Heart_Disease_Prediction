import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        background: linear-gradient(90deg, #ff8c00ff, #ff00f2ff);
        color: white;
        font-weight: bold;
        border-radius: 15px;
        border: none;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-safe {
        background: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .prediction-risk {
        background: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="background: linear-gradient(90deg, #ff8c00ff, #ff00f2ff); 
        font-size: 2.5rem; 
        font-weight: bold; 
        display: inline-block; 
        color: white; 
        text-align: center; 
        border-radius: 15px; 
        padding: 1rem 2rem;
        width: 100%;
        margin-bottom: 2rem;">
        ❤️ Heart Disease Prediction App
    </h1>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("heart_model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return model, scaler, label_encoders, feature_cols
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        return None, None, None, None

model, scaler, label_encoders, feature_cols = load_artifacts()

# Sidebar
with st.sidebar:
    st.markdown("""
        <h2 style="text-align: center; color: #ff8c00;">📋 Patient Information</h2>
        <p style="text-align: center; color: #666;">Enter the patient's medical details below</p>
    """, unsafe_allow_html=True)
    
    age = st.slider("Age", 20, 100, 75)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset Origin", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=180)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=400)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=80)
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.2, value=5.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    
    predict_button = st.button("🔍 Predict Heart Disease Risk")

if predict_button and model is not None:
    try:
        # Encode categorical variables
        encoded_data = {}
        encoded_data['sex'] = label_encoders['sex'].transform([sex])[0]
        encoded_data['dataset'] = label_encoders['dataset'].transform([dataset])[0]
        encoded_data['cp'] = label_encoders['cp'].transform([cp])[0]
        encoded_data['fbs'] = label_encoders['fbs'].transform([fbs])[0]
        encoded_data['restecg'] = label_encoders['restecg'].transform([restecg])[0]
        encoded_data['exang'] = label_encoders['exang'].transform([exang])[0]
        encoded_data['slope'] = label_encoders['slope'].transform([slope])[0]
        encoded_data['thal'] = label_encoders['thal'].transform([thal])[0]
        
        # Numeric values
        encoded_data['age'] = age
        encoded_data['trestbps'] = trestbps
        encoded_data['chol'] = chol
        encoded_data['thalch'] = thalach
        encoded_data['oldpeak'] = oldpeak
        encoded_data['ca'] = ca
        
        # Create DataFrame
        input_df = pd.DataFrame([encoded_data])[feature_cols]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                box_class = "prediction-box prediction-safe"
                emoji = "🟢"
                label = "No Heart Disease"
            else:
                box_class = "prediction-box prediction-risk"
                emoji = "🔴"
                label = "Heart Disease Detected"
            
            st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-size: 2.5rem;">{emoji}</div>
                    <div style="font-size: 1.8rem;">{label}</div>
                    <div style="font-size: 1rem; opacity: 0.8;">Confidence: {proba[prediction]*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=["No Disease", "Heart Disease"],
                    y=proba,
                    text=[f"{p:.1%}" for p in proba],
                    textposition="auto",
                    marker_color=["#28a745", "#dc3545"],
                    width=0.5,
                )
            ])
            fig.update_layout(
                title="Probability Distribution",
                yaxis=dict(title="Probability", range=[0, 1]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.markdown("### 📈 Identified Risk Factors")
        risk_factors = []
        if age > 60: risk_factors.append(f"Age: {age} years")
        if sex == "Male": risk_factors.append("Male gender")
        if cp in ["asymptomatic", "typical angina"]: risk_factors.append(f"Chest pain: {cp}")
        if trestbps > 140: risk_factors.append(f"High BP: {trestbps} mm Hg")
        if chol > 240: risk_factors.append(f"High cholesterol: {chol}")
        if thalach < 120: risk_factors.append(f"Low max HR: {thalach}")
        if exang == "True": risk_factors.append("Exercise induced angina")
        if oldpeak > 2.0: risk_factors.append(f"ST depression: {oldpeak}")
        if ca > 0: risk_factors.append(f"Major vessels: {ca}")
        if thal == "fixed defect": risk_factors.append("Fixed defect")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("No significant risk factors identified.")
            
        st.markdown("---")
        if prediction == 0:
            st.success("✅ Continue maintaining a healthy lifestyle!")
        else:
            st.error("🚨 Please consult a healthcare professional immediately!")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        <p>This application is for educational purposes only. Always consult with a healthcare professional for medical advice.</p>
    </div>
""", unsafe_allow_html=True)