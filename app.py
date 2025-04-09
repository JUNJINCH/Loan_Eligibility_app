import streamlit as st  # Streamlit for building the web interface
import pandas as pd  # Pandas for handling data in DataFrame format
import joblib  # Joblib for loading the trained model
from src import predict, preprocessing, visualization, logger  # Custom modules including logger

# Set the configuration for the Streamlit page
st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")

# Display the main title and description
st.markdown("""
# Loan Eligibility Predictor
This application estimates the likelihood of loan approval based on user-provided personal and financial information.
""")

# Form to collect user input
with st.form("loan_form"):
    st.markdown("### Personal Information")  # Section for personal details
    col1, col2 = st.columns([1, 1], gap="small")  # Two-column layout for personal info

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

    with col2:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    st.markdown("### Financial Information")  # Section for financial details
    col3, col4 = st.columns([1, 1], gap="small")  # Two-column layout for financial info

    with col3:
        applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Monthly Income", min_value=0, value=2000, step=100)
        loan_amount = st.number_input("Loan Amount ($1000)", min_value=0, value=150, step=10)

    with col4:
        loan_term = st.selectbox("Loan Amount Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360])
        credit_history = st.selectbox("Credit History", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Submit button
    submitted = st.form_submit_button("Predict Loan Eligibility")

# When the form is submitted
if submitted:
    try:
        # Create a DataFrame with user input
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [1.0 if credit_history == "Yes" else 0.0],  # Convert Yes/No to 1.0/0.0
            "Property_Area": [property_area]
        })

        # Log the input data
        logger.log_input(input_data)

        # Preprocess the input data
        df_processed = preprocessing.preprocess(input_data)

        # Make prediction using the loaded model
        prediction = predict.predict(df_processed)[0]

        # Log the prediction result
        logger.log_output(prediction)

        # Display prediction result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

        # Show feature importance section
        st.subheader("Feature Coefficient")
        st.markdown("""
A Logistic Regression model is used to predict the loan eligibility. 
The coefficients below show how much each feature influences the decision.
""")

        # Try to generate the feature importance plot
        try:
            fig = visualization.generate_feature_importance_plot(
                model_path="models/loan_model.pkl",
                feature_df=df_processed
            )
            st.pyplot(fig)
        except Exception as vis_err:
            st.error(f"Visualization failed: {vis_err}")
            logger.log_error(f"Visualization Error: {vis_err}")

    except Exception as e:
        st.error("An unexpected error occurred. Please check the input and try again.")
        logger.log_error(f"App Runtime Error: {e}")