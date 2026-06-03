import pandas as pd

from src.data_processing import (
    extract_time_features,
    create_customer_features
)


def test_extract_time_features():

    df = pd.DataFrame({
        "TransactionStartTime":
        ["2025-01-01 12:30:00"]
    })

    result = extract_time_features(df)

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns

def test_create_customer_features():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1"],
        "Amount": [100, 200],
        "Value": [100, 200]
    })

    result = create_customer_features(df)

    assert (
        result["transaction_count"]
        .iloc[0]
        == 2
    )

    assert (
        result["total_transaction_amount"]
        .iloc[0]
        == 300
    )