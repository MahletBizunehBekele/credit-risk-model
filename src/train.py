import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from mlflow import register_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

def load_data():

    df = pd.read_csv(
        "data/processed/model_data.csv"
    )

    return df

def prepare_data(df):

    X = df.drop(
        columns=["is_high_risk"]
    )

    y = df["is_high_risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = {
        "accuracy": accuracy_score(
                y_test,
                predictions
            ),

        "precision": precision_score(
                y_test,
                predictions
            ),

        "recall": recall_score(
                y_test,
                predictions
            ),

        "f1": f1_score(
                y_test,
                predictions
            ),

        "roc_auc": roc_auc_score(
                y_test,
                probabilities
            )
    }

    return metrics

def train_logistic(
    X_train,
    y_train
):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_random_forest(
    X_train,
    y_train
):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None]
    }

    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=42
        ),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        "Best Parameters:",
        grid.best_params_
    )

    return grid.best_estimator_

if __name__ == "__main__":

    df = load_data()

    print(df.columns.tolist())

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    mlflow.set_experiment(
        "credit_risk_model"
    )
        
            
    with mlflow.start_run(
            run_name="logistic_regression"
        ):

            logistic = train_logistic(
                X_train,
                y_train
            )

            metrics = evaluate_model(
                logistic,
                X_test,
                y_test
            )

            mlflow.log_params(
                {
                    "model":
                        "LogisticRegression"
                }
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.sklearn.log_model(
                logistic,
                "model"
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

            mlflow.register_model(
                model_uri=model_uri,
                name="CreditRiskModel"
            )
            print(metrics)

    with mlflow.start_run(
            run_name="random_forest"
        ):

            rf = train_random_forest(
                X_train,
                y_train
            )

            mlflow.log_params(
                rf.get_params()
            )

            metrics = evaluate_model(
                rf,
                X_test,
                y_test
            )

            mlflow.log_params(
                {
                    "model":
                        "RandomForest"
                }
            )

            joblib.dump(
                rf,
                "models/best_model.pkl"
            )
            mlflow.log_metrics(
                metrics
            )

            mlflow.sklearn.log_model(
                rf,
                "model"
            )

            print(metrics)
