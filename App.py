import streamlit as st
import pandas as pd
import pickle
import requests
from streamlit_extras.colored_header import colored_header

# Load model, scaler, and encoders
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("Standard_Scaler_mod.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="Income Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS 
st.markdown("""
<style>
    :root {
        --primary: #4F8BF9;
        --background: #121212;
        --secondary-bg: #1E1E1E;
        --text: #E0E0E0;
        --accent: #BB86FC;
        --success: #28a745;
        --danger: #dc3545;
    }
    
    body {
        background-color: var(--background);
        color: var(--text);
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
    }
    
    .st-b7, .st-c0, .st-ck {
        color: var(--primary) !important;
    }
    
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: var(--secondary-bg);
        color: var(--text);
        border-color: #444;
    }
    
    .stSlider>div>div>div>div {
        background-color: var(--primary);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #3a6fcc;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .st-expander {
        background-color: var(--secondary-bg);
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary);
    }
    
    .income-high {
        color: var(--success);
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .income-low {
        color: var(--danger);
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .profile-card {
        background-color: var(--secondary-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary-bg);
    }
    
    hr {
        border-color: #444;
    }
    
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border-color: var(--success) !important;
    }
    
    .stInfo {
        background-color: rgba(23, 162, 184, 0.1) !important;
        border-color: #17a2b8 !important;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.title("💡 Income Insights")
    
    st.markdown("""
    <div style='background-color: #2A2A2A; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h4 style='color: var(--success);'>Higher income likely if:</h4>
        <ul style='color: var(--text);'>
            <li>Age ≥ 35</li>
            <li>Education is Bachelor's or above</li>
            <li>Works > 40 hrs/week</li>
            <li>Capital gain/loss is non-zero</li>
            <li>Married</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #2A2A2A; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h4 style='color: var(--danger);'>Lower income likely if:</h4>
        <ul style='color: var(--text);'>
            <li>Age < 30</li>
            <li>Education is HS-grad or less</li>
            <li>Hours/week < 40</li>
            <li>Capital gain/loss = 0</li>
            <li>Not married</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Model Information", expanded=False):
        st.write("Model: Random Forest Classifier")
        st.write("Income classes:", label_encoders['income'].classes_)
        st.progress(87, text="Model Accuracy: 87%")

#Main Page
st.markdown("""
<div style='text-align: center; margin-top: 2rem; margin-bottom: 2rem;'>
    <h1 style='font-size: 3rem; color: #4F8BF9; font-weight: bold; letter-spacing: 1px;'>
        💰 INCOME PREDICTION DASHBOARD
    </h1>
    <p style='font-size: 1.3rem; color: #E0E0E0; margin-top: 0.5rem;'>
        Predict whether income exceeds $50K/yr based on census data
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='color: #AAAAAA;'>Enter your details below to get an income prediction and personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

def get_marital_status_binary(marital_status):
    if marital_status in ["Married-civ-spouse", "Married-AF-spouse"]:
        return 1
    else:
        return 0

#Input form
with st.form("user_info"):
    st.markdown("### 👤 PERSONAL & FINANCIAL INFORMATION")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 30, help="Select your age")
        sex = st.selectbox("Sex", label_encoders['sex'].classes_)
        education = st.selectbox("Education", label_encoders['education'].classes_, help="Select your highest education level")
        marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_, help="Select your marital status")
    with col2:
        occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_, help="Select your occupation")
        hours_per_week = st.slider("Hours per Week", 1, 100, 40, help="Typical hours worked per week")
        capital_gain = st.number_input("Capital Gain ($)", 0, 99999, 0, help="Income from capital gains")
        capital_loss = st.number_input("Capital Loss ($)", 0, 99999, 0, help="Losses from capital investments")

    submit_button = st.form_submit_button("🔮 PREDICT INCOME", use_container_width=True)

# Prepare data
data = {
    'age': age,
    'education': education,
    'marital_status': marital_status,
    'occupation': occupation,
    'sex': sex,
    'hours_per_week': hours_per_week,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'marital_status_binary': get_marital_status_binary(marital_status)
}
input_df = pd.DataFrame([data])

#Encode categorical features 
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = input_df[col].apply(
            lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0]
        )
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure correct column order
if hasattr(model, "feature_names_in_"):
    input_df = input_df[model.feature_names_in_]

# Scale features 
input_scaled = scaler.transform(input_df)

# Show user input
with st.expander("📋 REVIEW YOUR PROFILE", expanded=False):
    st.markdown("""
    <div class="profile-card">
        <h4 style='color: var(--primary); margin-bottom: 1rem;'>YOUR PROFILE SUMMARY</h4>
        <p><strong>Age:</strong> {age}</p>
        <p><strong>Education:</strong> {education}</p>
        <p><strong>Marital Status:</strong> {marital_status}</p>
        <p><strong>Occupation:</strong> {occupation}</p>
        <p><strong>Sex:</strong> {sex}</p>
        <p><strong>Hours/Week:</strong> {hours}</p>
        <p><strong>Capital Gain:</strong> ${gain:,}</p>
        <p><strong>Capital Loss:</strong> ${loss:,}</p>
    </div>
    """.format(
        age=age,
        education=education,
        marital_status=marital_status,
        occupation=occupation,
        sex=sex,
        hours=hours_per_week,
        gain=capital_gain,
        loss=capital_loss
    ), unsafe_allow_html=True)

# Predict and Query Gemini 
if submit_button:
    with st.spinner('Analyzing your profile and generating insights...'):
        try:
            pred = model.predict(input_scaled)[0]
            income_label = label_encoders['income'].inverse_transform([int(pred)])[0]
            
            if income_label == ">50K":
                st.markdown(f"""
                <div style='text-align: center; padding: 1.5rem; background-color: rgba(40, 167, 69, 0.1); border-radius: 10px; margin: 1rem 0; border-left: 4px solid var(--success);'>
                    <h3>🎯 PREDICTION RESULT</h3>
                    <p class="income-high">PREDICTED INCOME: {income_label}</p>
                    <p>Congratulations! Your profile suggests higher earning potential.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; padding: 1.5rem; background-color: rgba(220, 53, 69, 0.1); border-radius: 10px; margin: 1rem 0; border-left: 4px solid var(--danger);'>
                    <h3>🎯 PREDICTION RESULT</h3>
                    <p class="income-low">PREDICTED INCOME: {income_label}</p>
                    <p>See recommendations below to improve your earning potential.</p>
                </div>
                """, unsafe_allow_html=True)

            # Gemini API Integration 
            gemini_api_key = "AIzaSyAzTruaZJ4avtK40DSCA_ROWMCVGya7BUQ"  
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": gemini_api_key}
            
            prompt = (
                    f"The predicted income category for this individual is: {income_label}.\n\n"
                    f"User Profile:\n"
                    f"- Age: {input_df['age'].iloc[0]}\n"
                    f"- Education: {label_encoders['education'].inverse_transform([input_df['education'].iloc[0]])[0]}\n"
                    f"- Marital Status: {label_encoders['marital_status'].inverse_transform([input_df['marital_status'].iloc[0]])[0]}\n"
                    f"- Occupation: {label_encoders['occupation'].inverse_transform([input_df['occupation'].iloc[0]])[0]}\n"
                    f"- Sex: {label_encoders['sex'].inverse_transform([input_df['sex'].iloc[0]])[0]}\n"
                    f"- Hours/Week: {input_df['hours_per_week'].iloc[0]}\n"
                    f"- Capital Gain: {input_df['capital_gain'].iloc[0]}\n"
                    f"- Capital Loss: {input_df['capital_loss'].iloc[0]}\n\n"
                    f"Based on this profile and income prediction, provide helpful insights or recommendations "
                    f"for improving financial well-being or career advancement in **no more than 200 words**. "
                    f"Focus on realistic, actionable tips tailored to this individual."
                )


            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.9,
                    "maxOutputTokens": 300
                }
            }

            response = requests.post(gemini_url, headers=headers, params=params, json=payload)

            if response.status_code == 200:
                try:
                    gemini_reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                    
                    st.markdown("""
                    <div style='margin-top: 2rem;'>
                        <h3 style='color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem;'>🤖 PERSONALIZED RECOMMENDATIONS</h3>
                        <div style='background-color: var(--secondary-bg); padding: 1.5rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid var(--accent);'>
                            {gemini_reply}</div></div>
                    """.format(gemini_reply=gemini_reply), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("⚠️ Couldn't parse Gemini response. Here's the raw output:")
                    st.json(response.json())
            else:
                st.error("❌ Failed to get response from Gemini API.")
                st.code(response.text, language="json")

        except Exception as e:
            st.error("⚠️ An error occurred during prediction.")
            st.exception(e)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #6c757d; margin-top: 3rem;'>
    <p>Built by Muhammad Talha (2023-CS-12)<br>
    Assigned by M Kabir Ahmad and Prof. Talha Waheed</p>
    <p>For educational purposes only | Not financial advice</p>
</div>
""", unsafe_allow_html=True)