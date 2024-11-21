import torch
import numpy as np


# Define the EarlyStopping class, which is used to implement the early stopping mechanism during the training process
# to avoid overfitting and other situations.
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        """
        Function to initialize an instance of EarlyStopping.

        Parameters: - patience: The number of times to tolerate, that is, the number of epochs that training can
        continue when the validation loss has not improved. The default value is 5. - delta: The minimum improvement
        threshold. Only when the reduction in the validation loss is greater than this threshold is it considered an
        improvement. The default value is 0. - path: The file path to save the model. The default value is
        'checkpoint.pt'.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        # Used to record the number of times the validation loss has not improved.
        self.counter = 0
        # Used to record the best validation loss score so far, initialized as None.
        self.best_score = None
        # Used to mark whether the early stopping mechanism has been triggered, initialized as False.
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Allows an instance of the class to be called like a function. Here it is used to determine whether the early
        stopping conditions are met based on the current validation loss and other operations.

        Parameters:
        - val_loss: The current validation loss value.
        - model: The current model instance, which is used to save the model when the validation loss improves.
        """
        score = val_loss
        # Print the result of the judgment on whether the validation loss value is nan (not a number). Here,
        # it may be necessary to handle the nan situation better according to the actual situation later.
        print(str(score) == 'nan')

        if self.best_score is None:
            # If it is the first time to compare the validation loss, set the current validation loss as the best
            # validation loss and save the current model.
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif str(score) == 'nan' or score > self.best_score + self.delta:
            # If the current validation loss is nan or greater than the best validation loss plus the threshold (
            # meaning there is not enough improvement).
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # If the number of times without improvement reaches the tolerance number, trigger the early stopping
                # mechanism.
                self.early_stop = True
        else:
            # If the current validation loss has improved (less than the best validation loss plus the threshold).
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Function to save the model, which saves the current model to the specified file path.

        Parameters:
        - val_loss: The current validation loss value. Here it is used to print relevant prompt information when saving the model.
        - model: The current model instance, the object to be saved.
        """
        torch.save(model, self.path)
        print(f'Model saved! Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).')
