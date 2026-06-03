import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

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


def create_rfm_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x:
                (snapshot_date - x.max()).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Value",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm

def create_risk_labels(df):

    rfm = create_rfm_features(df)

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = (
        kmeans.fit_predict(rfm_scaled)
    )

    cluster_summary = (
        rfm.groupby("cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    print(cluster_summary)

    high_risk_cluster = (
        cluster_summary["Recency"]
        .idxmax()
    )

    rfm["is_high_risk"] = (
        rfm["cluster"] == high_risk_cluster
    ).astype(int)

    return rfm

def merge_risk_labels(df):

    rfm = create_risk_labels(df)

    df = df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    return df

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


# if __name__ == "__main__":

#     df = load_data(
#         "data/raw/data.csv"
#     )

#     processed_df = process_data(df)

#     processed_df.to_csv(
#         "data/processed/processed_data.csv",
#         index=False
#     )

#     print(processed_df.shape)

#     print(processed_df.head())
#     print(processed_df.columns.tolist())


if __name__ == "__main__":

    df = load_data("data/raw/data.csv")

    df = merge_risk_labels(df)

    print(df["is_high_risk"].value_counts())
    df.to_csv(
        "data/processed/labeled_data.csv",
        index=False
    )