import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split


df = pd.read_csv('bracketing_labeled_data.csv')

drop_cols = ['Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date', 'Return_Status']
df = df.drop(columns=drop_cols)

df = df.dropna()


label = 'is_bracketing'


train_data, test_data = train_test_split(
    df, test_size=0.2, stratify=df[label], random_state=42
)

predictor = TabularPredictor(label=label, path='model2_autogluon/')\
    .fit(
        train_data=train_data,
        time_limit=600, 
        presets='best_quality', 
        verbosity=2
    )


performance = predictor.evaluate(test_data)


leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard)

