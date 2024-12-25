from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from dataset import Datasets
from sklearn.metrics import r2_score

# Declare a global variable'model' which will be used to store the trained machine learning model later.
global model
# Set the name of the model to be used, here it's set to 'LightBGM'.
modelname = 'LightBGM'
# Call the 'Datasets' function (presumably defined elsewhere) to load the data and corresponding labels.
# The 'phnoe' parameter is set to 'HeadingDate' and the 'labelpath' points to a specific CSV file.
datas, labels = Datasets(type='snp', phnoe='HeadingDate', labelpath='frns_frnt_match.csv')
# Extract the first element of the 'labels' array. The exact reason for this depends on the data structure and what
# it represents.
label = labels[0]

# Read a file containing random indices, convert it to a flattened NumPy array.
# This will likely be used for shuffling or splitting the data in a random way.
random_permutation = np.array(pd.read_csv('random_index_HeadingDate.txt', header=None, sep='\t')).flatten()
# Reorder the 'label' array based on the random indices.
label = label[random_permutation]
# Reorder the 'data' array based on the random indices as well.
data = datas[random_permutation]

# Initialize empty lists to store various evaluation metrics and prediction results later.
all_r = []
all_pcc = []
y_all = []
out_all = []

# Loop 10 times, likely for a 10-fold cross-validation or similar splitting and evaluation process.
for j in range(1, 11):
    # Calculate the start and end indices for splitting the data into training and test sets for each fold.
    start_index = (j - 1) * 28
    end_index: int = j * 28

    # Concatenate parts of the data to form the training set, excluding the part designated for the test set in this
    # fold.
    train_data = np.concatenate((data[0:start_index], data[end_index:]))
    # Concatenate the corresponding parts of the labels to form the training labels.
    train_label = np.concatenate((label[0:start_index], label[end_index:]))

    # Extract the part of the data designated for the test set in this fold.
    test_data = data[start_index:end_index]
    # Extract the corresponding part of the labels for the test set.
    test_label = label[start_index:end_index]

    # Depending on the value of'modelname', create and train different machine learning models.
    if modelname == 'LightBGM':
        # Create an instance of the LightGBM Regressor with specified hyperparameters. Set early stopping rounds and
        # evaluation metric for training optimization. Then fit the model using the training data and labels,
        # also providing the test data and labels for evaluation during training.
        model = LGBMRegressor(n_estimators=100, max_depth=50, learning_rate=0.05)
        model.set_params(early_stopping_rounds=10, eval_metric='rmse')
        model.fit(train_data, train_label, eval_set=[(test_data, test_label)], )
    elif modelname == 'RF':
        # Create an instance of the Random Forest Regressor with specified hyperparameters and a fixed random state
        # for reproducibility. Train the model using the training data and labels.
        model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        model.fit(train_data, train_label)
    elif modelname == 'SVM':
        # Create an instance of the Support Vector Regression model with specified kernel, regularization parameter,
        # and tolerance. Train the model using the training data and labels.
        model = SVR(kernel='linear', C=1.0, epsilon=0.2)
        model.fit(train_data, train_label)
    elif modelname == 'RRBLUP':
        model = Ridge(alpha=0.1)
        model.fit(train_data, train_label-np.average(train_label))

    # Use the trained model to make predictions on the test data.
    y_pred = model.predict(test_data)
    if modelname == 'RRBLUP':
        y_pred = y_pred + np.average(train_label)
    # Calculate the R-squared score to evaluate how well the model's predictions match the actual test labels.
    r2 = r2_score(test_label, y_pred)
    # Calculate the Pearson correlation coefficient between the actual test labels and the predicted values.
    M = np.corrcoef(test_label, y_pred)
    p = M[0, 1]
    # Store the R-squared score for this fold in the 'all_r' list.
    all_r.append(r2)
    # Store the Pearson correlation coefficient for this fold in the 'all_pcc' list.
    all_pcc.append(p)

    # Concatenate the actual test labels and the predicted values to the respective lists for later overall analysis.
    y_all = np.concatenate((test_label, y_all))
    out_all = np.concatenate((y_pred, out_all))

# Append the average of the R-squared scores across all folds to the 'all_r' list.
all_r.append(np.average(all_r))
# Append the average of the Pearson correlation coefficients across all folds to the 'all_pcc' list.
all_pcc.append(np.average(all_pcc))

# Create a pandas DataFrame with columns 'y' (actual values) and 'out' (predicted values).
save = pd.DataFrame({'y': y_all, 'out': out_all})
# Save the DataFrame to a CSV file with a specific name based on the'modelname' and a '.svc' extension.
# Use tab as the delimiter and don't include row indices.
save.to_csv(f'HeadingDate_result_{modelname}.svc',
            sep='\t',
            index=False)
# Print the list of Pearson correlation coefficients for all folds and the average.
print(all_pcc)
# Create a pandas DataFrame with columns 'R2' (R-squared scores) and 'PCC' (Pearson correlation coefficients).
df = pd.DataFrame({"R2": all_r, 'PCC': all_pcc})
# Save this DataFrame to a text file with a specific name based on the'modelname' and a '.txt' extension.
# Use tab as the delimiter and don't include row indices.
df.to_csv(
    f'HeadingDate_result_{modelname}.txt',
    sep='\t',
    index=False)
