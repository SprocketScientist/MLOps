
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv("tourism_project/data/tourism.csv")

# Dropping unnecessary columns
df.drop(columns=["CustomerID","Gender"], inplace=True)

# Define target and features
y = df['ProdTaken']
X = df.drop('ProdTaken', axis=1)

# Identify categorical and numerical columns
categorical_features = ['TypeofContact', 'Occupation', 'ProductPitched', 'MaritalStatus', 'Designation']

# Ensure 'CityTier' and 'PitchSatisfactionScore' are included in numerical features if they are numerical
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a column transformer for one-hot encoding and standard scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features) # Apply StandardScaler to numerical features
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding for categorical features
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Get feature names after scaling for numerical features
scaled_feature_names = numerical_features

# Combine all feature names
all_feature_names = list(encoded_feature_names) + list(scaled_feature_names)

# Convert processed data back to DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

# Save the split datasets
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

print("Data preparation complete. Train and test sets saved.")
