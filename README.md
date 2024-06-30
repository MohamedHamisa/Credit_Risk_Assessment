# Credit Risk Prediction

This project aims to predict the credit risk of loan applicants using a RandomForestClassifier. The dataset used contains various features related to the applicants' personal and financial information.

## Table of Contents
- Installation
- Usage
- Data
- Model
- Evaluation
- Visualization
- Contributing
- License

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/credit-risk-prediction.git
    ```
2. Navigate to the project directory:
    ```
    cd credit-risk-prediction
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset:
    ```
    data = pd.read_csv('credit_risk_dataset.csv')
    ```
2. Run the data cleaning and feature engineering steps:
    ```
    data = data.dropna()
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])
    ```
3. Train the model and make predictions:
    ```
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
4. Evaluate the model:
    ```
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ```

## Data

The dataset contains the following columns:
- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_status`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

## Model

The model used is a RandomForestClassifier with 100 estimators and a random state of 42.

## Evaluation

The model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Visualization

The project includes the following visualizations:
- Confusion Matrix
- Feature Importance
- Pairplot

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


