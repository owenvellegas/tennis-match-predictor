from sklearn.model_selection import train_test_split
from src.model.xgboost import XGBoost
from src.model.evaluate_model import evaluate_model
import pandas as pd

def train_model(df: pd.DataFrame):

    y = df['win_loss'].values
    df = df.drop(columns='win_loss')
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X, y, X_test, y_test, X_train, y_train,)
    