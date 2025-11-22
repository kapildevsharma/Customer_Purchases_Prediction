# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/kapilmika/customer-purchases-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/kapilmika/customer-purchases-prediction/Xtest.csv"
ytrain_path = "hf://datasets/kapilmika/customer-purchases-prediction/ytrain.csv"
ytest_path = "hf://datasets/kapilmika/customer-purchases-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
# One-hot encode categorical features and scale numeric features
numeric_features = [
    'Age',               # Customer's age
    'CityTier',            # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'DurationOfPitch',          # Duration of the sales pitch delivered to the customer
    'NumberOfPersonVisiting',   # Number of people accompanying the customer on the trip.
    'NumberOfFollowups',        # Number of follow-ups by the salesperson after the sales pitch.
    'PreferredPropertyStar',    # Preferred hotel rating by the customer.
    'NumberOfTrips',    # Average number of trips the customer takes annually.
    'Passport',    # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore',    # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',    # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting',    # Number of children below age 5 accompanying the customer
    'MonthlyIncome'    # Customer's monthly income
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',  # The method by which the customer was contacted (Company Invited or Self Inquiry).
    'Occupation',     # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',         # Gender of the customer (Male, Female, Fe Male).
    'ProductPitched', # The type of product pitched to the customer.
    'MaritalStatus',  # Marital status of the customer (Single, Married, Divorced).
    'Designation'     # Customer's designation in their current organization.
]

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [75, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.5, 0.6],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all CV results
    results = grid_search.cv_results_
    # Save full CV results as a CSV (recommended)
    df = pd.DataFrame(results)
    df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Log best CV score
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

	  # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_customer_purchase_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "kapilmika/customer_purchases_model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_customer_purchase_model_v1.joblib",
        path_in_repo="best_customer_purchase_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
