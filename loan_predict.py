import streamlit as st
import pandas as pd
import joblib
model=joblib.load("load_model.pkl")
st.title("Loan Approval predictor")
st.write("Enter the following details to check if your loan is likely to be approved: ")
depedent=st.number_input("Number of Dependents",min_value=0,max_value=5)
education=st.selectbox("Education",['Graduate','Not Graduate'])
employed=st.selectbox("Employed?",['Yes','No'])
income_annum=st.number_input("Annual Income",min_value=0.0)
loan_amount=st.number_input("Loan Amount",min_value=0.0)
loan_term=st.number_input("Loan Term(years)",min_value=0.0)
cibil_score=st.number_input("Cibil score",min_value=300,max_value=850)
residential_assets_value=st.number_input("Residential Assets value",min_value=0.0)
commercial_assets_value=st.number_input("Commericial Assets value",min_value=0.0)
luxury_assets_value=st.number_input("Luxury Assets value",min_value=0.0)
bank_asset_value=st.number_input("Bank Assets value",min_value=0.0)

if st.button("predict"):
    data=pd.DataFrame({
        'no_of_dependents':[depedent],
        'education':[1 if education=='Graduate' else 0],
        'employed':[1 if employed=='Yes' else 0],
        'income_annum':[income_annum],
        'loan_amount':[loan_amount],
        'loan_term':[loan_term],
        'cibil_score':[cibil_score],
        'residential_assets_value':[residential_assets_value],
        'commercial_assets_value':[commercial_assets_value],
        'luxury_assets_value':[luxury_assets_value],
        'bank_asset_value':[bank_asset_value]
    })
    # columns_to_scale=scaler.transform(data[columns_to_scale])
    # make prediction
    prediction =model.predict(data)[0]
    probability=model.predict_proba(data)[0][1]
    if prediction==1:
        st.success(f"Loan Approved with {probability*100:.2f}% confidence.")
    else:
        st.error(f"Loan rejected with{100-probability*100:.2f}% confidence.")
    
