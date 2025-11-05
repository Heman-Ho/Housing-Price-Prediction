import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Helper function: Use when we know features and target -> for training 
def data_and_values(data, features):

    # Store feature for each entry
    mtx = []

    # store corresponding output 
    val = []

    # loop through each entry in the data
    for entry in data:

        # store the transformed value 
        phis = []

        for f, phi in features:

            # Apply transformed function to each feature value 
            phis.append(phi(entry[f]))

        mtx.append(phis)

        val.append(entry['price_ping'])

    return np.array(mtx).T, np.array(val).reshape(-1,1)

# Helper function: Use when we know features only, and we want to predict the output -> use for prediction
def data_only(data, features):

    # store feature for each entry
    mtx = []

    # loop through each entry in data
    for entry in data:

        phis = []

        for f, phi in features:

            # apply transformed function to each feature value 
            phis.append(phi(entry[f]))

        mtx.append(phis)

    return np.array(mtx).T

def standard(x, mean, std):

    return (x-mean)/std

def get_errors(y_test, y_predict): 
     # Calculates common regression error metrics between actual (y_test) and predicted values (y_predict).
     # Returns a dictionary containing:
     # - RMSE (Root Mean Squared Error)
     # - MAE (Mean Absolute Error)
     # - MSE (Mean Squared Error)
     rmse = np.sqrt(mean_squared_error(y_test, y_predict))

     mae = mean_absolute_error(y_test, y_predict)

     mse = mean_squared_error(y_test, y_predict)
     
     return {"RMSE": rmse, "MAE": mae, "MSE": mse}