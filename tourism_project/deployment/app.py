
import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "Wellness_Tourism_Predictor_model_v1.joblib")
model = joblib.load(model_path)

st.title("Visit with Us - Wellness Tourism Package: Customer Outcome Predictor")
st.write("""
This application predicts the likelihood of a customer purchasing the wellness tourism package based on the provided information.
""")

Age                     = st.number_input("Age",18,99,25,1)
TypeofContact           = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier                = st.selectbox("City Tier", [1,2,3])
DurationOfPitch         = st.number_input("Duration of Pitch",1,120,30,1)
Occupation              = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
MaritalStatus           = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfPersonVisiting  = st.number_input("Number of Person Visiting",1,10,2,1)
NumberOfFollowups       = st.number_input("Number of Followups",1,10,2,1)
ProductPitched          = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar   = st.number_input("Preferred Property Star",1,5,3,1)
NumberOfTrips           = st.number_input("Number of Trips",1,25,2,1)
Passport                = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore  = st.number_input("Pitch Satisfaction Score",1,5,3,1)
OwnCar                  = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting= st.number_input("Number of Children Visiting",0,10,0,1)
Designation             = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager","AVP", "VP"])
MonthlyIncome           = st.number_input("Monthly Income",1000,100000,10000,1)

# adding the input data into a pd dataframe
input_data = pd.DataFrame([{
    "Age" : Age,
    "TypeofContact" : TypeofContact,
    "CityTier" : CityTier,
    "DurationOfPitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "MaritalStatus" : MaritalStatus,
    "NumberOfPersonVisiting" : NumberOfPersonVisiting,
    "NumberOfFollowups" : NumberOfFollowups,
    "ProductPitched" : ProductPitched,
    "PreferredPropertyStar" : PreferredPropertyStar,
    "Marital Status" : MaritalStatus,
    "NumberOfTrips" : NumberOfTrips,
    "Passport" : Passport,
    "PitchSatisfactionScore" : PitchSatisfactionScore,
    "OwnCar" : OwnCar,
    "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "MonthlyIncome" : MonthlyIncome
}]) 

if st.button("Predict Outcome"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")    
