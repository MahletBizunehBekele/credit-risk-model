# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops a credit risk prediction system using alternative transaction data from the Xente dataset. Since the dataset does not contain an explicit default label, a proxy target variable is engineered using customer transaction behavior through RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.

---

## Credit Scoring Business Understanding

### Basel II Interpretability Requirements

Under Basel II regulatory frameworks, financial institutions must be able to explain and justify credit decisions. Credit risk models should therefore provide interpretable and auditable predictions that can be understood by stakeholders, regulators, and risk managers.

### Proxy Target Variable Necessity

The Xente dataset does not contain a direct loan default indicator. To enable supervised learning, a proxy target variable (`is_high_risk`) was created using customer transaction behavior. Customers with infrequent transactions, low monetary activity, and long inactivity periods were identified through RFM analysis and clustering and labeled as higher risk.

### Business Risks

Incorrect classification can lead to:

* False Positives: Creditworthy customers may be denied access to financial products.
* False Negatives: High-risk customers may receive credit and potentially default.
* Regulatory and reputational risks from poor model decisions.

### Interpretability vs. Predictive Performance Trade-off

Interpretable models such as Logistic Regression provide transparency and easier regulatory compliance. More complex models such as Random Forest often achieve higher predictive performance but are less transparent. This project evaluates both approaches and compares their performance.

---

## Project Structure

```text
src/
├── api/
├── data_processing.py
├── train.py
├── predict.py

tests/
├── test_data_processing.py
```

## Feature Engineering

* Aggregate customer features
* Datetime feature extraction
* Missing value handling
* One-hot encoding
* Standardization
* RFM analysis

## Model Training

Models implemented:

* Logistic Regression
* Random Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Hyperparameter tuning was performed using GridSearchCV.

## Experiment Tracking

MLflow was used to track:

* Parameters
* Metrics
* Model artifacts

## API Deployment

FastAPI provides:

* `/`
* `/predict`

Interactive documentation is available through Swagger UI.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/train.py
```

### Run Tests

```bash
pytest
```

### Start API

```bash
uvicorn src.api.main:app --reload
```

### Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```
