
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()


# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Small grid so the pipeline runs fast on GitHub Actions.
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring="recall", n_jobs=-1)
grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print(classification_report(ytest, best_model.predict(Xtest)))

# Save next to app.py so the Streamlit app can load it directly
joblib.dump(best_model, "deployment/best_machine_failure_model_v1.joblib")
print("Model saved to deployment/best_machine_failure_model_v1.joblib")
