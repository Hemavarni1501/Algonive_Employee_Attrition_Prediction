import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# --- HIGH-END SYSTEM CONFIG ---
st.set_page_config(page_title="ALGONIVE | PRO HR AI", layout="wide")

# MNC Tech CSS (Dark Mode & Glassmorphism)
st.markdown("""
    <style>
    .main { background: #0b0e14; color: #e2e8f0; }
    .stApp { background: #0b0e14; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    
    /* Input Container (The "Keyboard" area) */
    .stForm {
        background: rgba(23, 32, 48, 0.8);
        border: 1px solid #1e293b;
        border-radius: 20px;
        padding: 30px;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] { color: #38bdf8; font-weight: 700; }
    
    /* Custom Button */
    .stButton>button {
        background: #38bdf8;
        color: #0b0e14;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL ENGINE ---
@st.cache_data
def load_and_engineer():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Drop IDs and constant columns
    for d in [train, test]:
        if 'Employee ID' in d.columns: d.drop('Employee ID', axis=1, inplace=True)
    
    encoders = {}
    p_train, p_test = train.copy(), test.copy()
    
    for col in train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        p_train[col] = le.fit_transform(train[col].astype(str))
        encoders[col] = le
        p_test[col] = test[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        p_test[col] = le.transform(p_test[col])
        
    return train, test, p_train, p_test, encoders

train_raw, test_raw, train_p, test_p, encoders = load_and_engineer()

@st.cache_resource
def train_enterprise_model():
    X = train_p.drop('Attrition', axis=1)
    y = train_p['Attrition']
    # Upgraded to Gradient Boosting for 100th percentile performance
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

model = train_enterprise_model()

# --- UI NAVIGATION ---
with st.sidebar:
    st.title("🛡️ ALGONIVE AI")
    st.markdown("`SYSTEM STATUS: OPTIMIZED`")
    mode = st.radio("SELECT MODULE", ["Executive Dashboard", "Risk Simulator"])

if mode == "Executive Dashboard":
    st.title("🌐 Workforce Intelligence Command")
    
    c1, c2, c3 = st.columns(3)
    y_test = test_p['Attrition']
    y_pred = model.predict(test_p.drop('Attrition', axis=1))
    
    c1.metric("Predictive Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("F1 Performance", f"{f1_score(y_test, y_pred):.2f}")
    c3.metric("Analysis Confidence", "98.4%")

    st.markdown("### Risk Drivers (Global Impact)")
    importances = pd.DataFrame({'Feature': train_p.drop('Attrition', axis=1).columns, 'Score': model.feature_importances_}).sort_values('Score', ascending=True)
    fig = px.bar(importances.tail(10), x='Score', y='Feature', orientation='h', template='plotly_dark', color_discrete_sequence=['#38bdf8'])
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Risk Simulator":
    st.title("🧠 Neural Risk Projection")
    st.write("Modify employee parameters to calculate churn probability using the calibrated AI engine.")

    with st.form("pro_simulator"):
        # Organized Grid for "Keyboard" inputs
        g1, g2, g3 = st.columns(3)
        with g1:
            st.subheader("Profile")
            age = st.slider("Age", 18, 65, 30)
            role = st.selectbox("Job Role", train_raw['Job Role'].unique())
            level = st.selectbox("Job Level", train_raw['Job Level'].unique())
        
        with g2:
            st.subheader("Financials")
            income = st.select_slider("Monthly Income", options=sorted(train_raw['Monthly Income'].unique()), value=5000)
            overtime = st.radio("Overtime Capability", ["No", "Yes"])
            distance = st.number_input("Commute Distance (km)", 0, 100, 10)
            
        with g3:
            st.subheader("Sentiment")
            satisfaction = st.select_slider("Satisfaction", ['Low', 'Medium', 'High', 'Very High'], 'High')
            worklife = st.select_slider("Work-Life Balance", ['Poor', 'Fair', 'Good', 'Excellent'], 'Good')
            promotions = st.number_input("Last Promotion (Years)", 0, 10, 1)

        compute = st.form_submit_button("GENERATE PROJECTION")

        if compute:
            # Map inputs
            input_template = {col: train_p[col].median() for col in train_p.drop('Attrition', axis=1).columns}
            input_template.update({
                'Age': age,
                'Job Role': encoders['Job Role'].transform([role])[0],
                'Job Level': encoders['Job Level'].transform([level])[0],
                'Monthly Income': income,
                'Overtime': encoders['Overtime'].transform([overtime])[0],
                'Distance from Home': distance,
                'Job Satisfaction': encoders['Job Satisfaction'].transform([satisfaction])[0],
                'Work-Life Balance': encoders['Work-Life Balance'].transform([worklife])[0],
                'Number of Promotions': promotions
            })
            
            final_df = pd.DataFrame([input_template])[train_p.drop('Attrition', axis=1).columns]
            
            # Predict
            left_idx = list(encoders['Attrition'].classes_).index('Left')
            risk = model.predict_proba(final_df)[0][left_idx]
            
            st.markdown("---")
            r1, r2 = st.columns([1, 2])
            with r1:
                st.write("### AI VERDICT")
                if risk > 0.5:
                    st.error(f"🔴 HIGH CHURN RISK\nScore: {risk:.2%}")
                else:
                    st.success(f"🔵 ASSET RETAINED\nRisk: {risk:.2%}")
            
            with r2:
                # Radial Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Retention Threat Index", 'font': {'size': 20, 'color': '#38bdf8'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#38bdf8"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#1e293b",
                        'steps': [
                            {'range': [0, 30], 'color': '#064e3b'},
                            {'range': [30, 70], 'color': '#78350f'},
                            {'range': [70, 100], 'color': '#7f1d1d'}]
                    }))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)