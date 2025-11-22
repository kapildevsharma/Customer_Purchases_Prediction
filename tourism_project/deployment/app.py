import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="kapilmika/customer_purchases_model", filename="best_customer_purchase_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Customer Purchase Prediction")
st.write("""
The Customer Purchase Prediction App is an internal tool for the Visit with Us team that predicts
whether a customer is likely to purchase the new Wellness Tourism Package,
 helping staff prioritize outreach and improve marketing effectiveness.
""")

# User input
age = st.number_input("Customer's age", min_value=18, max_value=70, value=25, step=1)
city_tier = st.number_input("The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).", min_value=1, max_value=3, value=1, step=1)
duration_of_pitch = st.number_input("Total duration of the sales pitch delivered to the customer.", min_value=5, max_value=200, value=10, step=1)
number_of_person_visiting = st.number_input("Number of people accompanying the customer on the trip.", min_value=1, max_value=5, value=2, step=1)
number_of_followups = st.number_input("Number of follow-ups by the salesperson after the sales pitch.", min_value=1, max_value=6, value=2, step=1)
preferred_property_star = st.number_input("Preferred hotel rating by the customer.", min_value=3, max_value=5, value=4, step=1)
number_of_trips = st.number_input("Average number of trips the customer takes annually.", min_value=1, max_value=50, value=2, step=1)
passport = st.number_input("Whether the customer holds a valid passport (0: No, 1: Yes).", min_value=0, max_value=1, value=1, step=1)
pitch_satisfaction_score = st.number_input("Score indicating the customer's satisfaction with the sales pitch.", min_value=1, max_value=5, value=2, step=1)
own_car = st.number_input("Whether the customer owns a car (0: No, 1: Yes).", min_value=0, max_value=1, value=1, step=1)
number_of_children_visiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0, max_value=3, value=1, step=1)
monthly_income = st.number_input("Customer's monthly income", min_value=1000, max_value=100000, value=2500)

type_of_contact = st.selectbox("The method by which the customer was contacted (Company Invited or Self Inquiry).", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Customer's occupation (e.g., Salaried, Freelancer,Small Business, Large Business).", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
gender = st.selectbox("Gender of the customer (Male, Female).", ["Male", "Female", "Fe Male"])
product_pitched = st.selectbox("The type of product pitched to the customer.", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
marital_status = st.selectbox("Marital status of the customer (Single, Married, Divorced, Unmarried).", ["Single", "Married", "Divorced","Unmarried"])
designation = st.selectbox("Customer's designation in their current organization.", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'OwnCar': own_car,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'Age': age,
    'DurationOfPitch': duration_of_pitch,
    'Passport': passport,
    'CityTier': city_tier,
    'Gender': gender,
    'Occupation': occupation,
    'NumberOfTrips': number_of_trips,
    'NumberOfFollowups': number_of_followups,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'TypeofContact': type_of_contact,
    'MonthlyIncome': monthly_income,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'MaritalStatus': marital_status
}])

# Predict button
if st.button("Predict Customer Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **Customer has purchased a package : {result}**")
