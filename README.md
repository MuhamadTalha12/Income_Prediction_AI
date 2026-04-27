# 💰 Income Prediction Dashboard

A machine learning-powered web application that predicts whether an individual's income exceeds $50K/year based on census demographic data. Built with Streamlit, featuring real-time predictions, personalized recommendations powered by Google Gemini AI, and an elegant dark-themed UI.

## 🎯 Features

- **ML-Powered Predictions**: Random Forest classifier trained on census data with 87% accuracy
- **Interactive Dashboard**: User-friendly form to input demographic and financial information
- **AI Recommendations**: Google Gemini API integration for personalized financial insights based on predictions
- **Dark Theme UI**: Modern, custom-styled interface with responsive design
- **Profile Review**: Expandable section to review entered information before prediction
- **Insight Panel**: Sidebar with key factors affecting income prediction
- **Real-Time Feedback**: Dynamic result visualization with success/warning states

## 🌐 Live Demo

Try the application online without installation:

🔗 **[Live Demo - Income Prediction Dashboard](https://incomepredictionai.streamlit.app/)**

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 87%
- **Features**: Age, Education, Marital Status, Occupation, Sex, Hours/Week, Capital Gain, Capital Loss
- **Target Classes**: ≤$50K, >$50K

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuhamadTalha12/AI_Prediction.git
   cd AI_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare model files**
   Ensure the following pickle files are in the project directory:
   - `random_forest_model.pkl` - Trained Random Forest model
   - `Standard_Scaler_mod.pkl` - StandardScaler for feature normalization
   - `label_encoders.pkl` - Label encoders for categorical features

### Running the Application

```bash
streamlit run App.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage

1. **Enter Your Information**
   - Fill in personal details: Age, Sex, Education Level, Marital Status
   - Provide employment info: Occupation, Hours per Week
   - Input financial data: Capital Gain and Capital Loss

2. **Get Prediction**
   - Click the "🔮 PREDICT INCOME" button
   - View prediction result with visual feedback

3. **Review Insights**
   - Expand "REVIEW YOUR PROFILE" to see a summary of your inputs
   - Check "PERSONALIZED RECOMMENDATIONS" for AI-generated financial advice

## 📁 Project Structure

```
AI_Prediction/
├── App.py                          # Main Streamlit application
├── random_forest_model.pkl         # Trained ML model
├── Standard_Scaler_mod.pkl         # Feature scaler
├── label_encoders.pkl              # Categorical feature encoders
├── requirements.txt                # Python dependencies
├── How to run.txt                  # Instructions
└── README.md                       # This file
```

## 🛠️ Technologies Used

- **Frontend**: Streamlit, HTML/CSS
- **Machine Learning**: Scikit-learn (Random Forest, StandardScaler)
- **Data Processing**: Pandas, NumPy
- **AI Integration**: Google Gemini 1.5 Flash API
- **Serialization**: Pickle
- **HTTP Requests**: Requests library
- **UI Components**: streamlit-extras

## 🔧 Dependencies

See `requirements.txt` for all required packages:
- streamlit
- pandas
- scikit-learn
- requests
- streamlit-extras

## 📚 Key Factors for Higher Income

According to the model, income is more likely to exceed $50K if:
- ✅ Age ≥ 35
- ✅ Education: Bachelor's degree or above
- ✅ Works > 40 hours/week
- ✅ Non-zero capital gains/losses
- ✅ Married status

## 📝 Model Features Explained

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Individual's age in years |
| Education | Categorical | Highest level of education completed |
| Marital Status | Categorical | Current marital status |
| Occupation | Categorical | Type of occupation/job |
| Sex | Categorical | Gender |
| Hours per Week | Numeric | Typical hours worked per week |
| Capital Gain | Numeric | Income from capital gains (in $) |
| Capital Loss | Numeric | Losses from capital investments (in $) |

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not be used as financial advice. Income predictions are based on historical census data patterns and may not reflect individual circumstances or future earning potential.

## 👨‍💻 Author

- **Muhammad Talha** (2023-CS-12)
- Assigned by: M Kabir Ahmad and Prof. Talha Waheed

## 📄 License

This project is for educational purposes. Please refer to your institution's guidelines for usage and distribution.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## 📧 Contact

For questions or feedback, please reach out to the project author.

---

**Built with ❤️ for educational excellence**
