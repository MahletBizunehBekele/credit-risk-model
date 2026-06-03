import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_time_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df

def create_customer_features(df):

    agg = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),

            total_value=("Value", "sum"),
            avg_value=("Value", "mean")
        )
        .reset_index()
    )

    agg["std_transaction_amount"] = (
        agg["std_transaction_amount"]
        .fillna(0)
    )

    return agg

def engineer_features(df):

    df = extract_time_features(df)

    customer_features = create_customer_features(df)

    df = df.merge(
        customer_features,
        on="CustomerId",
        how="left"
    )

    return df

columns_to_drop = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]


numerical_features = [
    "Amount",
    "Value",

    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",

    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",

    "total_value",
    "avg_value"
]

categorical_features = [
    "CurrencyCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy"
]

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

def process_data(df):

    df = engineer_features(df)

    df = df.drop(
        columns=columns_to_drop
    )

    transformed = (
        preprocessor.fit_transform(df)
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    return processed_df


if __name__ == "__main__":

    df = load_data(
        "data/raw/data.csv"
    )

    processed_df = process_data(df)

    processed_df.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )

    print(processed_df.shape)

    print(processed_df.head())
    print(processed_df.columns.tolist())