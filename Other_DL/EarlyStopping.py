import torch
import numpy as np

# Define the EarlyStopping class, which is used to implement the early stopping mechanism.
# During the model training process, when the performance on the validation set no longer improves within a certain number of epochs,
# the training will be stopped early to avoid overfitting and save training time.
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        """
        Initialization function for setting up parameters related to the early stopping mechanism and initializing some internal state variables.

        Args:
            patience (int): The number of training epochs that the mechanism can tolerate when the validation performance stops improving. For example, if it's set to 5, it means that if the validation performance (such as validation loss or other metrics) doesn't improve for 5 consecutive training epochs, the early stopping mechanism may be triggered.
            delta (float): The threshold that defines the improvement of the model's performance. If the improvement is less than this threshold, it's considered that there's no real improvement. For example, if the decrease in the validation loss is less than this delta value, it won't be regarded as a true performance enhancement.
            path (str): The file path where the best model's weights will be saved. When the model achieves the best performance on the validation set so far, its weights will be saved to this specified path for later loading and use.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        # A counter used to record the consecutive number of epochs when the validation performance doesn't improve. It's initialized to 0.
        self.counter = 0
        # Used to record the best performance score (e.g., the minimum validation loss value) on the validation set so far. It's initialized to None and will be assigned a value during the first validation.
        self.best_score = None
        # A flag used to mark whether the early stopping mechanism has been triggered. It's initialized to False and will be set to True when the early stopping conditions are met.
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Allows an instance of the class to be called like a function, implementing the core logic of the early stopping judgment.
        After each evaluation on the validation set, the validation loss value (val_loss) and the current model (model) are passed in to update the early stopping-related states.

        Args:
            val_loss (float): The current validation loss value obtained on the validation set, which will be compared with the previous best loss value to determine whether there's an improvement in performance.
            model: The current model instance being trained. When there's an improvement in the validation performance, the weights of this model will be saved.
        """
        score = val_loss
        print(str(score) == 'nan')

        if self.best_score is None:
            # If the current best performance score hasn't been assigned yet (i.e., during the first validation evaluation),
            # set the current validation loss value as the best score and save the weights of the current model because it's the best situation at this time.
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif str(score) == 'nan' or score > self.best_score + self.delta:
            # If the current validation loss value is NaN (indicating an invalid number, which may occur due to calculation exceptions, etc.)
            # or is greater than the previous best score plus the delta value (usually we expect the loss to decrease, so a larger value means it has become worse),
            # it indicates that the validation performance hasn't improved or there's an abnormal situation. In this case, increment the counter for the number of epochs with no improvement and print the current count status.
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # If the consecutive number of epochs with no improvement reaches the set patience value, trigger the early stopping mechanism by setting self.early_stop to True, indicating that the training should be stopped.
                self.early_stop = True
        else:
            # If the current validation loss value is smaller than the previous best score (meaning there's an improvement in performance),
            # update the best score, save the weights of the current model with better performance, and reset the counter for the number of epochs with no improvement to 0.
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        A function to save the current model's weights, which saves the weights of the input model instance to the specified file path.

        Args:
            val_loss (float): The current validation loss value. When saving the model's weights, relevant information about the change in the validation loss (from the previous best loss to the current loss) will be printed.
            model: The model instance whose weights are to be saved. The torch.save function will be used to save its weights to the specified path.
        """
        torch.save(model, self.path)
        print(f'Model saved! Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).')