import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

from src.preprocessing import load_and_engineer_features

def train_model(AAPL_data):

    df = load_and_engineer_features(AAPL_data)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X = X.select_dtypes(include=["float64", "int64"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print(f"Model trained Successfully")
    print(f"Accuracy: {accuracy: 4f}")

    return model, accuracy

if __name__ == "__main__":

    csv_file = os.path.join("data", "AAPL_data.csv")
    model, acc = train_model(csv_file)

