from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib


# Define paths
input_path = "./df_data/data.pkl"
output_dir = "./df_data"
metrics_path = f"{output_dir}/metrics.csv"
predictions_path = f"{output_dir}/predictions.csv"
model_hgb_path = f"{output_dir}/model_hgb.joblib"
model_rf_path = f"{output_dir}/model_rf.joblib"
model_xgb_path = f"{output_dir}/model_xgb.joblib"

# Read pickle
df_pandas = pd.read_pickle(input_path)

# --- Define feature columns ---

# From previous encoding
property_type_cols = ['property_type_CONDO', 'property_type_MANUFACTURED', 
                    'property_type_SINGLE_FAMILY', 'property_type_APARTMENT','property_type_TOWNHOUSE']
print(f"\nproperty_type columns:")
print(property_type_cols)

# From previous encoding 
state_cols = [col for col in df_pandas.columns if col.startswith('state_')]
print(f"\nState Columns:")
print(state_cols)

# From previous kmeans clustering
cluster_cols = [col for col in df_pandas.columns if col.startswith('cluster_')]

feature_cols = [
    'bathroom_number', 'living_space_log', 'land_space_sqft_log',
    'latitude', 'longitude', 'living_space_per_bedroom', 'bedroom_number',

] + property_type_cols + state_cols + cluster_cols
print(f"\nFeature Columns:")
print(feature_cols)

# Define X and y
X = df_pandas[feature_cols]
y = df_pandas['log_price']

# Drop rows any remaining rows with NaN in X or y
print(f"Number of rows before dropping NaN: {len(X)}")
X_y = pd.concat([X, y], axis=1).dropna()
X = X_y[feature_cols]
y = X_y['log_price']
print(f"Number of rows after dropping NaN: {len(X)}")

# Train-test split
print(f"\nPerforming Train-test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN or inf in numeric columns only
numeric_cols = ['bathroom_number', 'living_space_log', 'land_space_sqft_log',
                'latitude', 'longitude', 'living_space_per_bedroom', 'bedroom_number'
                ]
X_train[numeric_cols] = X_train[numeric_cols].astype(float)
X_test[numeric_cols] = X_test[numeric_cols].astype(float)


# Cap numerical features at 95th percentile
for col in numeric_cols:
    cap_value = X_train[col].quantile(0.95)
    X_train[col] = X_train[col].clip(upper=cap_value)
    X_test[col] = X_test[col].clip(upper=cap_value)

# Scale features for HistGradientBoosting
print(f"\nScaling features for HistGradientBoosting...")
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Model 1: HistGradientBoosting -----
print(f"\nRunning HistGradientBoosting Regressor...")
model_hgb = HistGradientBoostingRegressor(random_state=42)
model_hgb.fit(X_train, y_train)
y_pred_log_hgb = model_hgb.predict(X_test)
y_pred_hgb = np.expm1(y_pred_log_hgb)
y_test_original = np.expm1(y_test)
rmse_hgb = mean_squared_error(y_test_original, y_pred_hgb) ** 0.5
r2_hgb = r2_score(y_test_original, y_pred_hgb)
print(f"HistGradientBoosting: RMSE={rmse_hgb:.2f}, R²={r2_hgb:.4f}\n")

# ----- Model 2: Random Forest -----
print(f"\nRunning Random Forest Regressor...")
model_rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_log_rf = model_rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_log_rf)
rmse_rf = mean_squared_error(y_test_original, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test_original, y_pred_rf)
print(f"Random Forest: RMSE={rmse_rf:.2f}, R²={r2_rf:.4f}")

# ----- Model 3: XGBoost (Tuned) -----
print(f"\nRunning Tuned XGBoost...")
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                                param_grid_xgb, cv=3, scoring='r2', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
model_xgb = grid_search_xgb.best_estimator_
y_pred_log_xgb = model_xgb.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log_xgb)
rmse_xgb = mean_squared_error(y_test_original, y_pred_xgb) ** 0.5
r2_xgb = r2_score(y_test_original, y_pred_xgb)
print(f"Tuned XGBoost: RMSE={rmse_xgb:.2f}, R²={r2_xgb:.4f}")
print("Best parameters:", grid_search_xgb.best_params_)

# ----- Ensemble Model -----
y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2
rmse_ensemble = mean_squared_error(y_test_original, y_pred_ensemble) ** 0.5
r2_ensemble = r2_score(y_test_original, y_pred_ensemble)
print(f"Ensemble: RMSE={rmse_ensemble:.2f}, R²={r2_ensemble:.4f}")


# ----- Save Outputs -----
# Save metrics
print(f"\nSaving metrics...")
metrics = pd.DataFrame({
    'Model': ['HistGradientBoosting', 'RandomForest', 'XGBoost', 'Ensemble'],
    'RMSE': [rmse_hgb, rmse_rf, rmse_xgb, rmse_ensemble],
    'R2': [r2_hgb, r2_rf, r2_xgb, r2_ensemble]
})
metrics.to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}")

# Save predictions
print(f"\nSaving Predictions...")
predictions = pd.DataFrame({
    'actual_price': y_test_original,
    'predicted_hgb': y_pred_hgb,
    'predicted_rf': y_pred_rf,
    'predicted_xgb': y_pred_xgb,
    'predicted_ensemble': y_pred_ensemble
}, index=y_test.index)
predictions.to_csv(predictions_path, index=False)
print(f"Saved predictions to {predictions_path}")

# Save models
print(f"\nSaving models...")
joblib.dump(model_hgb, model_hgb_path)
joblib.dump(model_rf, model_rf_path)
joblib.dump(model_xgb, model_xgb_path)
print(f"Saved models to {model_hgb_path}, {model_rf_path}, {model_xgb_path}")