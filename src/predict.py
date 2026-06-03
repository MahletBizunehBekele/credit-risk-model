import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)

EXPECTED_COLUMNS = model.feature_names_in_

def load_model():
    return joblib.load(MODEL_PATH)

def predict(data):

    df = pd.DataFrame([data])

    df = df.reindex(
        columns=EXPECTED_COLUMNS,
        fill_value=0
    )

    prediction = model.predict(df)

    return int(prediction[0])