import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
import demo 
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers # type: ignore 


# --------------------------------------------- Load the Data -------------------------------------------------------------------------
# Load the real_estate.csc data
dataframe = pd.read_csv("real_estate.csv")

# The value after the decimal is the corresponding month of transaction, calculated by ((dates - year) * 12) + 1 
X = dataframe['transaction_date'].values.reshape(-1,1) 

y = dataframe['price_ping'].values

# Plot the raw data
plt.figure(figsize=(12,6))
plt.scatter(X, y, color = 'blue', label = 'Actual price', alpha=0.2)
plt.xlabel("Transaction date (year.month)")
plt.ylabel("Price per ping (NT$/ping)")
plt.title("Housing Prices vs. Transaction Date")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --------------------------------------------- Milestone 1: Using KNN -------------------------------------------------------------------

# Preparing for feature preprocess 
features = {
    f: (dataframe[f].mean(), dataframe[f].std())
    for f in ['transaction_date', 'house_age', 'distance_to_mrt',
              'num_convenience_stores', 'latitude', 'longitude']
}

# Features for KNN 
features_knn = [
    (f, lambda x, m=features[f][0], s=features[f][1]: demo.standard(x, m, s))
    for f in features
]

# Features for MLP 
features_mlp = features_knn

# Convert our csv into a list of dictionaries with keys to loop over data entry easier
data = dataframe.to_dict(orient='records')

# Feature preprocessing
datas, values = demo.data_and_values(data, features_knn)

datas = datas.T 

y = values.flatten()
# print("X shape: ", datas.shape)
# print("y shape: ", y.shape)

# Use 10-fold cross validation to find best k that give us lowest MSE 
fold = KFold(n_splits=10, shuffle=True, random_state=42)

table = []

evaluation = []

for k in range(1,5):

    mse_result = []

    for train, test in fold.split(datas):

        X_train, X_test = datas[train], datas[test]
        y_train, y_test = y[train], y[test]

        knn = KNeighborsRegressor(n_neighbors=k)

        knn.fit(X_train,y_train)

        y_predict = knn.predict(X_test)

        mse = mean_squared_error(y_test, y_predict)
        mse_result.append(mse)

    avg_mse = np.mean(mse_result)

    table.append({
            "K": k,
            "MSE": avg_mse
    })

    evaluation.append(avg_mse) # store average MSE for each k

# Try to find the best k value
kValueTable = pd.DataFrame(table)

print("K values with MSE")
print(kValueTable)

best_k_idx = np.argmin(evaluation)
best_k = best_k_idx+1

print("\nBest k: ", best_k)
print(f"MSE for best k: ", evaluation[best_k_idx]) # find the k whose average MSE is lowest 

# Train the model using 80% of the data and test with 20%
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(datas, y, test_size=0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance')

knn.fit(X_train_knn, y_train_knn)

# Prediction
y_predict_knn = knn.predict(X_test_knn)

# Print the error metrics -> give us overall error of our prediction
errors = demo.get_errors(y_test_knn, y_predict_knn)

print("\nKNN METRICS EVALUATION\n")
print("Errors on test set: \n"
       f"RMSE: {errors['RMSE']:.2f} | MAE: {errors['MAE']:.2f} | MSE: {errors['MSE']:.2f}")

# Extract transaction dates and original rows for the KNN test set
transaction_dates = np.array([entry['transaction_date'] for entry in data])

# Use the same splitting seed as the model so indices align with y_test_knn
_, test_index = train_test_split(np.arange(len(transaction_dates)), test_size=0.2, random_state=42)

# Build KNN export DataFrame from the original dataframe rows corresponding to the test indices
knn_df = dataframe.iloc[test_index].reset_index(drop=True)

# Add actual / predicted prices and model label
knn_df = knn_df[['transaction_date', 'house_age', 'distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude']].copy()
knn_df['actual_price_ping'] = y_test_knn
knn_df['predicted_price_ping'] = y_predict_knn
knn_df['model'] = 'KNN'

# Export KNN predictions to CSV
knn_results = knn_df
knn_results.to_csv('knn_predictions.csv', index=False)

# Plot the reuslt 
plt.figure(figsize=(12,6))
plt.scatter(knn_results['transaction_date'], y_test_knn, color='blue', label='Actual Price', alpha=0.2)
plt.scatter(knn_results['transaction_date'], y_predict_knn, color='red', label='Predicted Price', alpha=0.2, marker='x')
plt.xlabel("Transaction Date (Year.Month)")
plt.ylabel("Price per Ping (NT$/ping)")
plt.title("KNN Predicted Prices vs Actual Prices Over Time (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------- Milestone 1: Using KNN -------------------------------------------------------------------

# --------------------------------------------- Milestone 2: Using Decision Tree ---------------------------------------------------------

datas, values = demo.data_and_values(data, features_mlp)

datas = datas.T

y = values.flatten()

fold = KFold(n_splits=10, shuffle=True, random_state=42)

depths = range(2, 11)

mse_list = []

for depth in depths:

    fold_mse = []

    for train_idx, test_idx in fold.split(datas):

        X_train, X_test = datas[train_idx], datas[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DecisionTreeRegressor(max_depth=depth, random_state=42)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        fold_mse.append(mean_squared_error(y_test, preds))

    avg_mse = np.mean(fold_mse)

    mse_list.append(avg_mse)

    print(f"Depth {depth}: MSE = {avg_mse:.4f}")

best_depth = depths[np.argmin(mse_list)]

print(f"\nBest max_depth: {best_depth}, MSE: {min(mse_list):.4f}")

# Trains a final model on an 20/80 train-test split using the best max_depth
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(datas, y, test_size=0.2, random_state=42)

final_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)

final_model.fit(X_train_DT, y_train_DT)

y_pred_DT = final_model.predict(X_test_DT)


# Print the error metrics -> give us overall error of our prediction
errors = demo.get_errors(y_test_DT, y_pred_DT)

print("DECISION TREE METRICS EVALUATION")
print("Errors on test set: \n"
       f"RMSE: {errors['RMSE']:.2f} | MAE: {errors['MAE']:.2f} | MSE: {errors['MSE']:.2f}")

# Plots predicted vs actual prices with respect to transaction date.
test_dates = dataframe['transaction_date'].values[train_test_split(np.arange(len(dataframe)), test_size=0.2, random_state=42)[1]]

# Build Decision Tree export DataFrame from original rows for the test split
_, test_index_dt = train_test_split(np.arange(len(dataframe)), test_size=0.2, random_state=42)
dt_df = dataframe.iloc[test_index_dt].reset_index(drop=True)

dt_df = dt_df[['transaction_date', 'house_age', 'distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude']].copy()
dt_df['actual_price_ping'] = y_test_DT
dt_df['predicted_price_ping'] = y_pred_DT
dt_df['model'] = 'Decision Tree'

# Export Decision Tree predictions to CSV
dt_results = dt_df
dt_results.to_csv('dt_predictions.csv', index=False)

# Combine for model comparison
combined_results = pd.concat([knn_results, dt_results], ignore_index=True)
combined_results.to_csv('model_comparison_results.csv', index=False)

plt.figure(figsize=(12,6))
plt.scatter(test_dates, y_test_DT, color='blue', label='Actual Price', alpha=0.2)
plt.scatter(test_dates, y_pred_DT, color='red', label='Predicted Price', marker='x', alpha=0.2)
plt.xlabel("Transaction Date")
plt.ylabel("Price per Ping (NT$/ping)")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: visualize tree
plt.figure(figsize=(20,10))
plot_tree(final_model, filled=True, feature_names=[f[0] for f in features])
plt.title("Decision Tree Structure")
plt.show()

# --------------------------------------------- Milestone 2: Using Decision Tree ----------------------------------------------------------

# --------------------------------------------- Milestone 2: Using MLP --------------------------------------------------------------------

# Feature preprocessing
datas, values = demo.data_and_values(data, features_mlp)

datas = datas.T

y = values.flatten()

# Train the model using 80% of our data and test with 20% 
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(datas, y, test_size=0.2, random_state=42)

# Pass the features into 2 hidden layers, then learns the weight using optimization algorithm then train over 
# 110 iterations with 20 batch at a time 
algorithm = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(1e-4)), # L2 regularization
    Dropout(0.2),

    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.2),

    Dense(1)
])

algorithm.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Model training
algorithm.fit(X_train_mlp, y_train_mlp, validation_split=0.2, epochs=110, batch_size=20, verbose=1)

# Prediction
y_predict_mlp = algorithm.predict(X_test_mlp, verbose = 0).flatten()

# Print the error metrics -> give us overall error of our prediction
errors = demo.get_errors(y_test_mlp, y_predict_mlp)

print("MLP METRICS EVALUATION")
print("Errors on test set: \n"
       f"RMSE: {errors['RMSE']:.2f} | MAE: {errors['MAE']:.2f} | MSE: {errors['MSE']:.2f}")

# Extract transaction date
transaction_dates = np.array([entry['transaction_date'] for entry in data])

_, test_index = train_test_split(np.arange(len(transaction_dates)), test_size=0.2, random_state=42)

test_dates = transaction_dates[test_index]

# Plot the result
plt.figure(figsize=(12,6))
plt.scatter(test_dates, y_test_mlp, color='blue', label='Actual Price', alpha=0.2)
plt.scatter(test_dates, y_predict_mlp, color='red', label='Predicted Price', alpha=0.2, marker='x')
plt.xlabel("Transaction Date (Year.Month)")
plt.ylabel("Price per Ping (NT$/ping)")
plt.title("MLP Predicted Prices vs Actual Prices Over Time (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Future dates generation
# Fixed reference value for features except for transaction date
values = {f: features[f][0] for f in features if f != 'transaction_date'}

# Simulate how time progresses
future_dates = np.linspace(2014, 2019, 20).reshape(-1,1)

# Isolating time as only driver of the model
future_datas = []

for dates in future_dates:

    future_datas.append({
        'transaction_date': dates[0],
        'house_age': values['house_age'],
        'distance_to_mrt': values['distance_to_mrt'],
        'num_convenience_stores': values['num_convenience_stores'],
        'latitude': values['latitude'],
        'longitude': values['longitude']
    })

# Feature preprocessing -> for future datas 
X_future = demo.data_only(future_datas, features_mlp)

X_future = X_future.T

# Prediction
y_future_predict = algorithm.predict(X_future).flatten()

future_dates = future_dates.reshape(-1,1)

# Plot the result
plt.figure(figsize=(12,6))
plt.plot(future_dates, y_future_predict, label='Predicted future price', marker="o", color='red')
plt.xlabel("Transaction Date (Year.Month)")
plt.ylabel("Price per Ping (NT$/Ping)")
plt.title("MLP Future Housing Price Prediction")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------- Milestone 2: Using MLP ----------------------------------------------------------