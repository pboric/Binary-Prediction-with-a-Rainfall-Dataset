
# Rainfall Prediction Project

This project aims to predict rainfall based on meteorological data. It includes data exploration, preprocessing, feature engineering, model training, and evaluation.

## Project Structure

```
Rainfall Prediction/
├── data/
│   ├── train.csv         # Original training data
│   ├── test.csv          # Original test data
│   ├── train_processed.csv # Processed training data
│   └── test_processed.csv  # Processed test data
├── Rainfall_Prediction_EDA.ipynb         # Exploratory Data Analysis notebook
├── Rainfall_Prediction_Preprocessing.ipynb # Data Preprocessing and Feature Engineering notebook
├── Rainfall_Prediction_Modeling.ipynb    # Model Training and Evaluation notebook
├── best_model.pkl        # Serialized best trained model
└── submission.csv        # Submission file with predictions
```

## Dependencies

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `xgboost`
-   `lightgbm`
-   `joblib`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

## Project Workflow

1.  **Exploratory Data Analysis (EDA):**
    -      The `Rainfall_Prediction_EDA.ipynb` notebook performs an in-depth analysis of the dataset, including:
        -      Basic information and missing value checks.
        -      Target variable distribution analysis.
        -      Feature distribution analysis.
        -      Feature relationships with the target variable.
        -      Correlation analysis.
        -      Comparison of train and test data distributions.
        -      Feature range analysis.
        -      Outlier detection and analysis using Isolation Forest and Local Outlier Factor (LOF).
    -      This step helps understand the data's characteristics and identify potential issues.

2.  **Data Preprocessing and Feature Engineering:**
    -      The `Rainfall_Prediction_Preprocessing.ipynb` notebook handles data cleaning and feature engineering, including:
        -      Handling missing values.
        -      Removing unnecessary features (e.g., 'id').
        -      Outlier detection and handling (capping).
        -      Encoding cyclical features ('day', 'winddirection').
        -      Transforming skewed distributions.
        -      Creating interaction features.
        -      Handling multicollinearity.
        -      Handling bimodal distributions.
        -      Feature scaling.
    -      The processed data is saved as `train_processed.csv` and `test_processed.csv` in the `data/` directory.

3.  **Model Training and Evaluation:**
    -      The `Rainfall_Prediction_Modeling.ipynb` notebook trains and evaluates several machine learning models, including:
        -      Logistic Regression.
        -      Random Forest.
        -      XGBoost.
        -      LightGBM.
    -      It uses GridSearchCV with StratifiedKFold cross-validation to find the best hyperparameters for each model.
    -      Model performance is evaluated using ROC AUC.
    -   Feature importance is analyzed.
    -      The best performing model is selected, retrained on the full training dataset, and saved as `best_model.pkl`.
    -      Predictions are made on the test set, and a submission file (`submission.csv`) is generated.

## Running the Notebooks

1.  Clone the repository.
2.  Install the required dependencies using pip.
3.  Place the original `train.csv` and `test.csv` files in the `data/` directory.
4.  Run the notebooks in the following order:
    -   `Rainfall_Prediction_EDA.ipynb`
    -   `Rainfall_Prediction_Preprocessing.ipynb`
    -   `Rainfall_Prediction_Modeling.ipynb`

## Saved Model

The best trained model is saved as `best_model.pkl` using `joblib`. You can load and use it for predictions as follows:

```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('best_model.pkl')

# Load test data
test = pd.read_csv('data/test_processed.csv')
original_test = pd.read_csv('data/test.csv')
test_ids = original_test['id']

# Make predictions
predictions = model.predict_proba(test)[:, 1]

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'rainfall': predictions})
submission.to_csv('submission_prediction.csv', index=False)
```

## Submission

The final predictions are saved in `submission.csv`, which can be used for further analysis.
