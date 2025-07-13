import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np



df = pd.read_csv("bracketing_labeled_data.csv")
drop_cols = ['Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date', 'Return_Status']
df = df.drop(columns=drop_cols)
df = df.dropna()

print(f"Total samples: {len(df)}")
print(f"Bracketing cases: {(df['is_bracketing'] == 1).sum()}")
print(f"Non-bracketing cases: {(df['is_bracketing'] == 0).sum()}")
print(f"Bracketing percentage: {(df['is_bracketing'] == 1).mean() * 100:.2f}%")


X = df.drop(columns=['is_bracketing'])
y = df['is_bracketing']


X = pd.get_dummies(X)


feature_columns = X.columns
joblib.dump(feature_columns, "bracketing_feature_columns.pkl")


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set bracketing cases: {y_train.sum()}")
print(f"Test set bracketing cases: {y_test.sum()}")


class_weights = {
    0: 1.0,
    1: 5.0
}

print(f"Class weights: {class_weights}")


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values(ascending=False).head(10))


print("\nTesting specific cases...")

test_case_1 = pd.DataFrame([{
    'Product_Price': 450,
    'Order_Quantity': 4,
    'Days_to_Return': 15,
    'User_Age': 30,
    'Discount_Applied': 30,
    'Product_Category_Clothing': 1,
    'User_Gender_Female': 1,
    'Payment_Method_Credit Card': 1,
    'Shipping_Method_Express': 1,
    'Size_L': 1 
}])


for col in feature_columns:
    if col not in test_case_1.columns:
        test_case_1[col] = 0

test_case_1 = test_case_1[feature_columns]
pred_1 = model.predict(test_case_1)[0]
prob_1 = model.predict_proba(test_case_1)[0]

print(f"Test Case 1 (High-risk clothing): Prediction={pred_1}, Probability={prob_1}")


test_case_2 = pd.DataFrame([{
    'Product_Price': 100,
    'Order_Quantity': 1,
    'Days_to_Return': -1,
    'User_Age': 40,
    'Discount_Applied': 10,
    'Product_Category_Electronics': 1,
    'User_Gender_Male': 1,
    'Payment_Method_Credit Card': 1,
    'Shipping_Method_Standard': 1
}])

for col in feature_columns:
    if col not in test_case_2.columns:
        test_case_2[col] = 0

test_case_2 = test_case_2[feature_columns]
pred_2 = model.predict(test_case_2)[0]
prob_2 = model.predict_proba(test_case_2)[0]

print(f"Test Case 2 (Low-risk electronics): Prediction={pred_2}, Probability={prob_2}")


test_case_3 = pd.DataFrame([{
    'Product_Price': 200,
    'Order_Quantity': 2,
    'Days_to_Return': -1,
    'User_Age': 35,
    'Discount_Applied': 15,
    'Product_Category_Books': 1,
    'User_Gender_Male': 1,
    'Payment_Method_Credit Card': 1,
    'Shipping_Method_Standard': 1
}])


for col in feature_columns:
    if col not in test_case_3.columns:
        test_case_3[col] = 0

test_case_3 = test_case_3[feature_columns]
pred_3 = model.predict(test_case_3)[0]
prob_3 = model.predict_proba(test_case_3)[0]

print(f"Test Case 3 (Medium-risk books): Prediction={pred_3}, Probability={prob_3}")

test_case_4 = pd.DataFrame([{
    'Product_Price': 400,
    'Order_Quantity': 5,
    'Days_to_Return': 10,
    'User_Age': 25,
    'Discount_Applied': 20,
    'Product_Category_Clothing': 1,
    'User_Gender_Female': 1,
    'Payment_Method_PayPal': 1,  # PayPal is common in bracketing
    'Shipping_Method_Express': 1,
    'Size_S': 1,
    'Size_M': 1,
    'Size_L': 1,
    'Size_XL': 1
}])


for col in feature_columns:
    if col not in test_case_4.columns:
        test_case_4[col] = 0

test_case_4 = test_case_4[feature_columns]
pred_4 = model.predict(test_case_4)[0]
prob_4 = model.predict_proba(test_case_4)[0]

print(f"Test Case 4 (Very high-risk clothing): Prediction={pred_4}, Probability={prob_4}")


joblib.dump(model, "bracketing_model.pkl")
print("\nModel saved successfully!")
