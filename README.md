# Credit Risk Probability Model for Alternative Data

This project develops an end-to-end credit risk scoring system for Bati Bank using transaction data from an eCommerce platform. The objective is to construct a proxy credit risk target, engineer predictive features, train and evaluate machine learning models, and deploy the best model as a production-ready API.

Credit Scoring Business Understanding
1. Basel II and Model Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory compliance in credit risk assessment. Financial institutions must be able to explain how a credit decision was made and demonstrate that their models are reliable, stable, and properly validated.

Because credit decisions affect customers directly and are subject to regulatory review, models must be interpretable and well documented. Clear documentation allows auditors, regulators, and business stakeholders to understand the assumptions, features, and decision logic used by the model. Interpretability also supports model monitoring, validation, and risk management throughout the model lifecycle.

For this project, Basel II considerations encourage the use of explainable features, reproducible modeling pipelines, and thorough documentation of all modeling decisions.

2. Need for a Proxy Target Variable

The provided dataset contains transaction records but does not contain a direct indicator of customer default. Since supervised machine learning requires labeled examples, a proxy target variable must be created to represent credit risk.

A proxy variable can be constructed using customer behavioral patterns such as Recency, Frequency, and Monetary (RFM) metrics. Customers with low engagement, infrequent transactions, and low spending activity may be considered higher-risk customers and can be used as a proxy for potential default behavior.

However, proxy-based prediction introduces business risks. The proxy may not perfectly represent actual default events, leading to labeling errors and model bias. Customers classified as high-risk by the proxy may not actually default, while some future defaulters may be labeled as low-risk. These inaccuracies can affect lending decisions, customer experience, and portfolio performance. Therefore, the limitations of the proxy target must be clearly documented and monitored.

3. Trade-offs Between Interpretable and High-Performance Models

There is an important trade-off between model interpretability and predictive performance in credit risk modeling.

Logistic Regression combined with Weight of Evidence (WoE) transformation is highly interpretable. Individual feature contributions can be explained easily, making the model suitable for regulatory environments and easier to validate. However, its predictive power may be limited when relationships between variables are complex or nonlinear.

More advanced models such as Gradient Boosting, XGBoost, or LightGBM often achieve higher predictive accuracy because they can capture nonlinear patterns and interactions between variables. However, these models are more difficult to explain and may require additional interpretability techniques such as SHAP values or feature importance analysis.

In regulated financial environments, institutions often prioritize transparency and regulatory compliance while still seeking strong predictive performance. The final model choice should balance explainability, accuracy, operational requirements, and regulatory expectations.