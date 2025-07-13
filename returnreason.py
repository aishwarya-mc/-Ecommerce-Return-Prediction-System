import joblib
import pandas as pd


model = joblib.load("return_reason_model.pkl")
scaler = joblib.load("scaler.pkl")
le_reason = joblib.load("label_encoder_reason.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def predict_return_reason(user_input_dict):
 
    input_df = pd.DataFrame([user_input_dict])

    input_df['Total_Amount'] = input_df['Product_Price'] * input_df['Order_Quantity']


    if 'Days_to_Return' not in input_df:
        input_df['Days_to_Return'] = -1

 
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)


    input_df[['Product_Price', 'Discount_Applied', 'Total_Amount']] = scaler.transform(
        input_df[['Product_Price', 'Discount_Applied', 'Total_Amount']]
    )

    
    pred_label = model.predict(input_df)[0]
    return le_reason.inverse_transform([pred_label])[0]
