import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import demo  # uses demo.py for preprocessing

#Load data
df = pd.read_csv("real_estate.csv")
data = df.to_dict(orient="records")

#Define features
features = [
    ('transaction_date', demo.standard),
    ('house_age', demo.standard),
    ('distance_to_mrt', demo.standard),
    ('num_convenience_stores', demo.standard),
    ('latitude', demo.standard),
    ('longitude', demo.standard)
]

# Preprocess
X_all, y_all = demo.data_and_values(data, features)
X_all = X_all.T
y_all = y_all.flatten()

#Performs 10-fold cross-validation to find the best max_depth for the decision tree.
fold = KFold(n_splits=10, shuffle=True, random_state=42)
depths = range(2, 11)
mse_list = []
mae_list = []
rmse_list = []


for depth in depths:
    fold_mse = []
    fold_mae = []
    fold_rmse = []
    for train_idx, test_idx in fold.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors = demo.get_errors(y_test, preds)
        fold_mse.append(errors['MSE'])
        fold_mae.append(errors['MAE'])
        fold_rmse.append(errors['RMSE'])
    avg_mse = np.mean(fold_mse)
    avg_mae = np.mean(fold_mae)
    avg_rmse = np.mean(fold_rmse)
    mse_list.append(avg_mse)
    mae_list.append(avg_mae)
    rmse_list.append(avg_rmse)
    print(f"Depth: {depth} | RMSE = {avg_rmse:.2f} | MAE = {avg_mae:.2f} | MSE = {avg_mse:.2f}")

best_depth = depths[np.argmin(mse_list)]
print(f"\nBest max_depth (according to MSE): {best_depth} | MSE: {min(mse_list):.2f} | MAE: {min(mae_list):.2f} | RMSE: {min(rmse_list):.2f}")

#Trains a final model on an 80/20 train-test split using the best max_depth
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
final_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

#Evaluates the model on test data using Mean Squared Error (MSE).
test_errors = demo.get_errors(y_test, y_pred)
print(f"Test Errors (Decision Tree) | RMSE: {test_errors['RMSE']:.2f} | MAE: {test_errors['MAE']:.2f} | MSE: {test_errors['MSE']:.2f}")

#Plots predicted vs actual prices with respect to transaction date.
test_dates = df['transaction_date'].values[train_test_split(np.arange(len(df)), test_size=0.5, random_state=42)[1]]

plt.figure(figsize=(12,6))
plt.scatter(test_dates, y_test, color='blue', label='Actual Price', alpha=0.5)
plt.scatter(test_dates, y_pred, color='red', label='Predicted Price', marker='x', alpha=0.5)
plt.xlabel("Transaction Date")
plt.ylabel("Price per Ping (NT$/ping)")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Optional: visualize tree
plt.figure(figsize=(20,10))
plot_tree(final_model, filled=True, feature_names=[f[0] for f in features])
plt.title("Decision Tree Structure")
plt.show()
