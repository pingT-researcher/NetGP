import torch
import time
from predict import Predict


def train(mymodel, epoch, optimizer, scheduler, criterion, device, trainloader, testloader, early_stopping):
    """
    This function is used to train the given model. It iterates over a specified number of epochs, performing forward
    and backward passes, updating the model's parameters, and also includes functionality for early stopping and
    evaluating the model's performance on a test set.

    Parameters: - mymodel: The neural network model to be trained. - epoch: The number of training epochs,
    indicating how many times the entire training dataset will be iterated over. - optimizer: The optimization
    algorithm (e.g., Adam, SGD, etc.) used to update the model's parameters to minimize the loss. - scheduler: A
    learning rate scheduler that adjusts the learning rate during the training process according to a certain
    strategy. - criterion: The loss function used to measure the difference between the model's predictions and the
    ground truth labels. - device: The device (e.g., 'cuda' for GPU or 'cpu' for CPU) on which the model and data
    will be processed. - trainloader: The data loader for the training dataset, which provides batches of input data,
    corresponding labels, and potentially other related information. - testloader: The data loader for the test
    dataset, used to evaluate the model's performance during training at certain intervals. - early_stopping: An
    instance of an early stopping mechanism that monitors the validation loss and decides whether to stop the
    training early to prevent overfitting.
    """
    mymodel.train()
    """
    Set the model to training mode. This is important as it enables certain behaviors like dropout (if used) and batch normalization updates
    that are specific to the training phase.
    """
    start = time.time()
    for i in range(epoch):
        all_loss = 0
        for (inputs, labels, trans) in trainloader:
            torch.cuda.empty_cache()
            """
            Clear the GPU cache.
            """
            inputs = inputs.to(device)
            labels = labels.to(device)
            trans = trans.to(device)
            """
            Move the input data, labels, and other related tensors
            """
            optimizer.zero_grad()
            """
            Clear the gradients of all the model's parameters. 
            """
            outs = mymodel(inputs, trans)

            outs = outs.reshape(-1)
            loss = criterion(outs, labels)
            """
            Compute the loss between the model's predictions and the ground truth labels using the specified loss function.
            """
            all_loss += loss.item()
            loss.backward()
            """
            Perform backpropagation to compute the gradients of the loss with respect to the model's parameters.
            """
            optimizer.step()
            """
            Update the model's parameters using the computed gradients and the optimization algorithm 
            """
        scheduler.step()
        """
        Update the learning rate according to the defined learning rate scheduler.
        """
        print(i, "train_loss===>", all_loss / len(trainloader))
        with torch.no_grad():
            """
            Disable gradient calculation as we are only evaluating the model and don't need to compute gradients during this part.
            """
            if i != 0 and i % 5 == 0:
                mymodel.eval()
                r2, p, y, out, val_loss = Predict(device, mymodel, criterion, testloader)
                early_stopping(val_loss, mymodel)
                if early_stopping.early_stop:
                    """Check if the early stopping condition has been met. If so, print a message and break out of 
                    the training loop to stop further training."""
                    print("Early stopping")
                    break
                mymodel.train()
    end = time.time()
    # 打印模型的权重
    # if first!=True:
    #     torch.save(mymodel, model_file)
    print(mymodel)

    # 检查所有层的权重
    # for name, param in mymodel.named_parameters():
    #     print(name, param.data)
    # print('snp_mask_weight',mymodel.snp_lin.mask_weight.data)
    # print('gene_mask_weight',mymodel.gene_lin.mask_weight.data)
    # print('pathway_mask_weight',mymodel.pathway_lin.mask_weight.data)
    print(end - start)
    """
    Print the total time taken for the entire training process.
    """
