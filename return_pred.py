import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


order_time_features = [
    'Product_Category', 'Product_Price', 'Order_Quantity', 'User_Age', 'User_Gender',
    'User_Location', 'Payment_Method', 'Shipping_Method', 'Discount_Applied'
]
df = pd.read_csv('ecommerce_returns_synthetic_data_realistic.csv')


if 'Return_Status' not in df.columns:
    raise ValueError('Return_Status column missing!')
df = df.dropna(subset=['Return_Status'])


df = df[order_time_features + ['Return_Status']]

for col in ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = ['Product_Category', 'User_Gender', 'User_Location', 'Payment_Method', 'Shipping_Method']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


df['Return_Status'] = df['Return_Status'].map({'Returned': 1, 'Not Returned': 0})


scaler = StandardScaler()
numeric_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop('Return_Status', axis=1)
y = df['Return_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_get_important_features(X, y, label, importance_threshold=0.01):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    features = X.columns
    sorted_idx = importances.argsort()[::-1]


    print(f"\n Classification Report ({label}):")
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [features[i] for i in sorted_idx], rotation=90)
    plt.title(f"Feature Importances ({label})")
    plt.tight_layout()
    plt.savefig("featurn_imp_grapg.png")
    plt.close()


    important = pd.Series(importances, index=features)
    return important[important > importance_threshold].index.tolist()


important_cols = train_and_get_important_features(X_train, y_train, "All Features")


X_train_imp = X_train[important_cols]
X_test_imp = X_test[important_cols]

model = RandomForestClassifier(random_state=42)
model.fit(X_train_imp, y_train)

y_pred = model.predict(X_test_imp)

print("\n Classification Report (Important Features Only on Test Set):")
print(classification_report(y_test, y_pred))


pd.Series(important_cols).to_csv("important_features.csv", index=False)
print("\n Saved important feature names to 'important_features.csv'")
