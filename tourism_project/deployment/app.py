
import os
import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Tourism Package Predictor", layout="centered")

# Load model and preprocessor artifacts
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "Wellness_Tourism_Predictor_model_v1.joblib")
preprocessor_path = os.path.join(current_dir, "preprocessor.joblib")

@st.cache_resource
def load_artifacts():
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(preprocessor_path)
    return loaded_model, loaded_preprocessor

model, preprocessor = load_artifacts()

st.title("Visit with Us - Wellness Tourism Package: Customer Outcome Predictor")
st.write(
    "This application predicts the likelihood of a customer purchasing the wellness tourism package based on user demographics and interaction features."
)

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 99, 25, 1)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (mins)", 1, 120, 30, 1)
    Occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2, 1)
    NumberOfFollowups = st.number_input("Number of Followups", 1, 10, 2, 1)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3, 1)
    NumberOfTrips = st.number_input("Number of Trips", 1, 25, 2, 1)
    Passport_str = st.selectbox("Passport", ["Yes", "No"])
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
    OwnCar_str = st.selectbox("Own Car", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 10, 0, 1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", 1000, 1000000, 20000, 1000)

# Map binary selections to numeric 1/0
Passport = 1 if Passport_str == "Yes" else 0
OwnCar = 1 if OwnCar_str == "Yes" else 0

if st.button("Predict Outcome"):
    # Construct raw DataFrame with schema matching prep.py input
    raw_input_df = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])

    # Transform raw inputs using fitted preprocessor
    processed_array = preprocessor.transform(raw_input_df)

    # Reconstruct DataFrame with feature names expected by XGBoost
    cat_features = ['TypeofContact', 'Occupation', 'ProductPitched', 'MaritalStatus', 'Designation']
    encoded_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
    num_names = [col for col in raw_input_df.columns if col not in cat_features]
    all_feature_names = encoded_names + num_names

    processed_df = pd.DataFrame(processed_array, columns=all_feature_names)

    # Inference
    prediction = model.predict(processed_df)[0]
    probabilities = model.predict_proba(processed_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Outcome: **Product Likely Taken** (Confidence: {probabilities[1]:.2%})")
    else:
        st.info(f"Outcome: **Product Not Likely Taken** (Confidence: {probabilities[0]:.2%})")
