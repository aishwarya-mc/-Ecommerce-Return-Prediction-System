from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split


predictor = TabularPredictor.load("model2_autogluon/")

df = pd.read_csv("bracketing_labeled_data.csv")
drop_cols = ['Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date', 'Return_Status']
df = df.drop(columns=drop_cols)
df = df.dropna()



train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['is_bracketing'], random_state=42)

y_test = test_data['is_bracketing']
X_test = test_data.drop(columns=['is_bracketing'])

preds = predictor.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(" Accuracy:", accuracy_score(y_test, preds))
print("\n Classification Report:\n", classification_report(y_test, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
