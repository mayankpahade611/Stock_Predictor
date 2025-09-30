import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

from src.preprocessing import load_and_engineer_features

from xgboost import XGBClassifier

def train_model(TSLA_data):

    df = load_and_engineer_features(TSLA_data)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X = X.select_dtypes(include=["float64", "int64"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

   

    # model = RandomForestClassifier((n_estimators=100, random_state=42, class_weight="balanced")
    # model.fit(X_train, y_train)
    model = XGBClassifier(
        n_estimators = 200,
        learning_rate = 0.05,
        max_depth = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        eval_metric = "logloss",
        scale_pos_weight = 18/31,
    )

   

    model.fit(X_train, y_train)

 

    

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print(f"Model trained Successfully")
    print(f"Accuracy: {accuracy: 4f}")

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    print(model.feature_importances_)

    return model, accuracy

if __name__ == "__main__":

    csv_file = os.path.join("data", "TSLA_data.csv")
    model, acc = train_model(csv_file)

