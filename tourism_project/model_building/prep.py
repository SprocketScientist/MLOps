
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Load data
df = pd.read_csv("tourism_project/data/tourism.csv")

# 2. Drop identifiers, sensitive features, and index artifacts safely
cols_to_drop = [col for col in ["CustomerID", "Gender", "Unnamed: 0"] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# 3. Separate features and target
y = df['ProdTaken']
X = df.drop('ProdTaken', axis=1)

# 4. Train/Test split BEFORE transformation to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define column groups
categorical_features = ['TypeofContact', 'Occupation', 'ProductPitched', 'MaritalStatus', 'Designation']
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 6. Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# 7. Fit ONLY on train data, transform both train and test
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

# 8. Reconstruct feature names
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(encoded_feature_names) + list(numerical_features)

X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

# 9. Export datasets for train.py
X_train_df.to_csv('Xtrain.csv', index=False)
X_test_df.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

# 10. Save the fitted preprocessor for inference in app.py
output_dir = "tourism_project/deployment"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

print("Data preparation complete. Datasets and preprocessor.joblib saved successfully.")
