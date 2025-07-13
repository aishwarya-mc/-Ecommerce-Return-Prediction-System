import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


df = pd.read_csv('ecommerce_returns_synthetic_data.csv')  

df = df.drop('User_Location', axis=1)


df['Return_Status'] = df['Return_Status'].map({'Returned': 1, 'Not Returned': 0})


df['Days_to_Return'] = df['Days_to_Return'].fillna(-1)
df['Return_Reason'] = df['Return_Reason'].fillna('No Return')


df = df[df['Days_to_Return'] >= -1]


df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Order_Weekday'] = df['Order_Date'].dt.weekday
df['Order_Month'] = df['Order_Date'].dt.month
df['Order_Year'] = df['Order_Date'].dt.year


df = df.drop(['Order_ID', 'Product_ID', 'User_ID', 'Return_Date', 'Order_Date'], axis=1)


df['Total_Amount'] = df['Product_Price'] * df['Order_Quantity']


categorical_cols = ['Product_Category', 'User_Gender',
                    'Payment_Method', 'Shipping_Method', 'Return_Reason']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


df.to_csv('preprocessed_dataset.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_dataset.csv'")


scaler = StandardScaler()
df[['Product_Price', 'Discount_Applied', 'Total_Amount']] = scaler.fit_transform(
    df[['Product_Price', 'Discount_Applied', 'Total_Amount']]
)


X = df.drop('Return_Status', axis=1)
y = df['Return_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(" Classification Report:\n")
print(classification_report(y_test, y_pred))
