import numpy as np
import torch
from sklearn.metrics import r2_score


# Define a function named 'Predict' which is used to make predictions using the provided model on the test dataset
# and calculate several evaluation metrics related to the model's performance.
def Predict(device, mymodel, criterion, testloader):
    """
    Function: Predict Description: This function takes in a device (e.g., 'cuda' or 'cpu'), a trained model (
    mymodel), a loss criterion (criterion), and a test data loader (testloader). It performs predictions on the test
    data using the model, calculates the loss, and computes some common evaluation metrics such as R-squared (R2) and
    correlation coefficient.

    Parameters: - device: The device (like 'cuda' for GPU or 'cpu' for CPU) where the tensors will be processed
    during the prediction process. - mymodel: The trained neural network model that will generate predictions for the
    input data from the testloader. - criterion: The loss function that measures the difference between the predicted
    outputs and the actual labels. Commonly used loss functions include Mean Squared Error (MSE), etc. - testloader:
    The data loader for the test dataset, which iteratively provides batches of input data, corresponding labels,
    and potentially other auxiliary information (in this case, 'trans') for each batch.

    Returns: - r2: The R-squared value which indicates how well the model's predictions fit the actual data. - p: The
    correlation coefficient between the actual labels and the predicted outputs. - y: The list of actual labels from
    the test dataset. - out: The list of predicted outputs from the model for the test dataset. - all_loss / len(
    testloader): The average loss over all the batches in the testloader, which reflects the model's performance in
    terms of the chosen loss function.
    """
    all_loss = 0
    y = []
    out = []
    # Iterate over each batch of data provided by the testloader.
    for (inputs, labels, trans) in testloader:
        tmp_label = labels.cpu().detach().numpy().flatten()
        y = np.concatenate((y, tmp_label))
        labels = labels.to(device)
        inputs = inputs.to(device)
        trans = trans.to(device)
        # Use the provided model (mymodel) to generate predictions for the current batch's input data.
        outs = mymodel(inputs, trans)
        outs = outs.reshape(-1)
        # Calculate the loss for the current batch using the provided loss criterion (criterion).
        # The loss measures how different the predicted outputs are from the actual labels.
        loss = criterion(outs, labels)
        all_loss += loss.item()
        tmp_out = outs.cpu().detach().numpy().flatten()
        out = np.concatenate((out, tmp_out))
    print("Loss==>", all_loss / len(testloader))
    print(y)
    print(out)
    # Calculate the R-squared (R2) value using the sklearn's r2_score function. R2 measures the proportion of the
    # variance in the dependent variable (actual labels) that is predictable from the independent variable (predicted
    # outputs).
    r2 = r2_score(y, torch.tensor(out, dtype=torch.float64))
    # Calculate the correlation coefficient matrix between the actual labels and the predicted outputs using NumPy's
    # corrcoef function. The correlation coefficient indicates the strength and direction of the linear relationship
    # between two variables.
    M = np.corrcoef(y, torch.tensor(out, dtype=torch.float64))
    # Extract the correlation coefficient value from the correlation coefficient matrix. In the resulting 2x2 matrix,
    # the element at position [0, 1] (or [1, 0] as it's symmetric) represents the correlation between the two variables.
    p = M[0, 1]
    print("R21==>", r2)
    print("P1==>", p)
    return r2, p, y, out, all_loss / len(testloader)
