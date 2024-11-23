import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

try:
    # Load your dataset
    data = pd.read_csv("loan_eligibility_prediction_dataset.csv")
    
    # Check if required columns exist
    if 'Loan_Status' not in data.columns:
        raise ValueError("Dataset does not contain the required 'Loan_Status' column.")
    
    # Assume 'X' contains your features, and 'y' contains the target variable
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    
    # Check for empty datasets
    if X.empty or y.empty:
        raise ValueError("Dataset is empty or missing data.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(include='number').columns
    categorical_columns = X.select_dtypes(exclude='number').columns

    # Create separate transformers for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Apply the preprocessor to training and testing sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Build and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_processed, y_train)

    # Take input from the user
    print("Enter details to check loan eligibility:")
    try:
        user_input = {
            "Loan_ID": "LP001002",  # Placeholder value or drop this line if not needed
            "Gender": input("Gender (Male/Female): "),
            "Married": input("Married (Yes/No): "),
            "Dependents": input("Dependents (Number of dependents): "),
            "Education": input("Education (Graduate/Not Graduate): "),
            "Self_Employed": input("Self Employed (Yes/No): "),
            "ApplicantIncome": float(input("Applicant Income: ")),
            "CoapplicantIncome": float(input("Coapplicant Income: ")),
            "LoanAmount": float(input("Loan Amount: ")),
            "Loan_Amount_Term": input("Loan Amount Term (in months): "),
            "Credit_History": float(input("Credit History (1.0 for Yes, 0.0 for No): ")),
            "Property_Area": input("Property Area (Urban/Rural/Semiurban): "),
            "Defaulted": input("Defaulted(Yes/No): ")
        }
    except ValueError as e:
        raise ValueError("Invalid input type. Please enter numeric values for numeric fields.") from e

    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Apply the preprocessor to user input
    try:
        user_input_processed = preprocessor.transform(user_df)
    except NotFittedError as e:
        raise RuntimeError("Preprocessor has not been fitted. Check your pipeline.") from e

    # Make prediction for the user
    try:
        prediction = model.predict(user_input_processed)
    except NotFittedError as e:
        raise RuntimeError("Model has not been fitted. Check your training process.") from e

    # Display the result
    if prediction[0] == 'Y':
        print("Congratulations! You are eligible for a loan.")
    else:
        print("Sorry, you are not eligible for a loan.")

except FileNotFoundError:
    print("Error: The dataset file 'loan_prediction_dataset.csv' was not found. Please check the file path.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
