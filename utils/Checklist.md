
Data Ingestion & Initial Inspection
* Understand the Objective & Load the data
* Quick EDA (Exploratory Data Analysis): Shape, types, and basic statistics.

Data Preprocessing & Cleaning
* Handle missing values
* Encode categorical variables
* Detect outliers

Feature Engineering
* Create new features that might be more predictive than the raw data.

Model Training & Validation
* Train a diverse set of "classical" ML models using a robust validation strategy (e.g., TimeSeriesSplit, Cross-Validation)
* Speed and comparability
* Store models and results efficiently

Model Evaluation & Diagnostics
* Compare models on relevant metrics (e.g., Sharpe Ratio, Precision/Recall, Mean Squared Error)
* Analyze residuals, feature importance, and look for overfitting.

Presentation & Visualization
* Create clear, insightful plots and a concise summary of your findings. Tell a story with the data.


## VERY IMPORTANT

fit_transform(X_train), NEVER fit(X_test), **only transform**(X_test)
