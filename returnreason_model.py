import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  


df = pd.read_csv("ecommerce_returns_synthetic_data.csv")
df = df[df['Return_Status'] == 'Returned']
df = df[df['Return_Reason'].notna()]


df = df.drop(['Order_ID', 'Product_ID', 'User_ID', 'Return_Date', 'Order_Date', 'User_Location'], axis=1)


le_reason = LabelEncoder()
df['Return_Reason_Label'] = le_reason.fit_transform(df['Return_Reason'])


df['Total_Amount'] = df['Product_Price'] * df['Order_Quantity']
df['Days_to_Return'] = df['Days_to_Return'].fillna(-1)


categorical_cols = ['Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
df[['Product_Price', 'Discount_Applied', 'Total_Amount']] = scaler.fit_transform(
    df[['Product_Price', 'Discount_Applied', 'Total_Amount']]
)

X = df.drop(['Return_Reason', 'Return_Reason_Label', 'Return_Status'], axis=1)
y = df['Return_Reason_Label']

feature_columns = X.columns


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_reason.classes_))


joblib.dump(model, "return_reason_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_reason, "label_encoder_reason.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
