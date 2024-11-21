import numpy as np
import pandas as pd
import torch
from torch import optim
# Import relevant functions and classes from custom modules. These modules are presumably defined by the user
# to handle tasks like dataset processing, prediction, training, etc.
from DataSet import Datasets, divide_Dataset
from predict import Predict
from train import train
from DeepGS import DeepGS
from DNNGP import DNNGP
from GenNet import GenNet
from EarlyStopping import EarlyStopping
# Set the device to 'cuda:0' if a CUDA-enabled GPU is available, otherwise use 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the default tensor type to torch.DoubleTensor. This means that tensors created later in the code
# will be of this type by default if no other type is specified.
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    #HeadingDate  DNNGP  snp    lr=1e-3     bs = 24   stps=25
    #HeadingDate  DNNGP  trans  lr=1e-3   bs = 16   stps=35
    #FT10   DNNGP  trans snp    lr=1e-3   bs = 24  stps = 25

    #HeadingDate  DeepGS  snp  trans  lr=1e-3     bs = 16   stps=30
    #FT10   DeepGS  trans snp    lr=1e-3   bs = 24   stps = 25

    #HeadingDate  GenNet snp  trans  lr=1e-3     bs = 24   stps=30
    #FT10   GenNet       snp    lr=1e-3   bs = 32   stps = 25
    
    # Define a variable representing the phenotype. It might correspond to a certain biological feature or trait.
    # In this context, it could be used to identify specific groups of related data in the dataset.
    phenos = 'HeadingDate'
    # Specify the name of the model to be used. Different model names correspond to different model architectures
    # and implementations. The code will instantiate the corresponding model object based on this name later.
    modelname = 'DeepGS'
    # Path to the SNP data file. Presumably, this file stores Single Nucleotide Polymorphism (SNP) data related to the study.
    # The format of the file should meet the requirements of the subsequent reading function (e.g., pd.read_csv).
    snp_file = f'PHeadingDate.txt'
    # Path to the label file. This file contains the label information corresponding to the dataset, which is used
    # to measure the accuracy of the model's predictions during the training and evaluation phases.
    label_file = f'phenos_match.txt'
    # Path to the gene-related data file. It's initialized as an empty string here, but in the case of specific models
    # like GenNet, it might be used to read and process additional gene-related information.
    gene_re = 'snp_gene_HeadingDate.txt'
    # Indicate the type of the dataset as'snp', meaning the data being processed is of the Single Nucleotide Polymorphism type.
    # Different data types may have different ways of being read, processed, and applied to models later.
    types ='snp'
    if modelname == 'GenNet':
        """
        If the model name is 'GenNet', call the custom 'Datasets' function to obtain the data, labels, and gene index information.
        The 'Datasets' function should be defined in the corresponding 'DataSet' module and is used to retrieve and process
        data from the appropriate files based on the given parameters.
        """
        datas, labels, genes_index = Datasets(types, snp_file, label_file, gene_re=gene_re)
        print('genes_index', genes_index.shape)
        print(genes_index)
    else:
        """
        If the model name is not 'GenNet', just call the 'Datasets' function to obtain the data and labels,
        as there's no need to handle the gene index information for other models.
        """
        datas, labels = Datasets(types, snp_file, label_file)
    print('data:', datas.shape)
    print('labels', labels.shape)
    # Select relative subscripts based on different trait files
    label = labels[0]
    # These indices will be used to reorder the data later.
    random_index_file = f'random_index_{phenos}.txt'
    random_permutation = np.array(pd.read_csv(random_index_file, sep='\t')).flatten()
    data = datas[random_permutation]
    label = label[random_permutation]

    # Define the loss function as the L1 Loss (Mean Absolute Error). This function measures the difference between
    # the model's predictions and the true labels. During the training process, the goal is to minimize this loss.
    criterion = torch.nn.L1Loss()
    # Initialize an empty list to store a certain evaluation metric
    all_r = []
    # Initialize another empty list to store another evaluation metric (e.g., Pearson Correlation Coefficient)
    all_pcc = []
    # Initialize an empty numpy array to store all the true label values. This array will be concatenated with new
    # true label values obtained in each iteration later.
    y_all = np.array([])
    # Initialize an empty numpy array to store all the model's predicted values. Similar to the true label array,
    # it will be concatenated with new predicted values obtained in each iteration.
    out_all = np.array([])
    for num in range(1, 11):
        # Set the learning rate, which controls the step size for updating the model's parameters during training.
        # A smaller value like 1e-3 is chosen here.
        lr = 1e-3
        # Set the batch size, which determines the number of data samples passed to the model in each training iteration.
        bach_size = 16
        # Set the step size for the learning rate scheduler. This means that every'step_size' epochs, the learning rate
        # will be adjusted according to a certain rule.
        step_size = 30
        """
        Call the 'divide_Dataset' function to split the data into training and test sets. Based on the provided parameters
        (the current iteration number 'num', the number of samples per fold '18', the batch size, and the model name),
        the function will return DataLoader objects for the training and test sets, which are convenient for loading
        data in batches for training and testing later.
        """
        trainloader, testloader = divide_Dataset(data, label, num, 18, bach_size, modelname)
        if modelname == 'DNNGP':
            """
            If the model name is 'DNNGP', instantiate the DNNGP model object and move it to the specified device (GPU or CPU).
            The parameter passed (len(data[0])) is likely related to the dimension of the data features and is used to
            initialize the model's structure.
            """
            mymodel = DNNGP(len(data[0])).to(device)
        elif modelname == 'DeepGS':
            """
            If the model name is 'DeepGS', instantiate the DeepGS model object and move it to the specified device.
            The passed parameter is also related to the data feature dimension and is used for initializing the model.
            """
            mymodel = DeepGS(len(data[0])).to(device)
        else:
            """
            For other models (mainly referring to GenNet in this context), instantiate the GenNet model object and move
            it to the specified device. The 'genes_index' parameter is passed for initializing the model since the GenNet
            model might rely on gene-related feature representations in its structure or computation.
            """
            mymodel = GenNet(genes_index).to(device)
        # Define the path where the model file will be saved. The file name includes the phenotype name and the model name,
        # which will be used to save the trained model or load a previously saved model later.
        model_file = f'/data2/users/xuzhi/tp/{phenos}_{modelname}.model'
        """
        Instantiate the EarlyStopping class to implement the early stopping mechanism during the training process.
        The 'patience' parameter indicates how many epochs the validation loss can remain unchanged before stopping
        the training. The 'delta' parameter represents the minimum improvement in the loss required for considering
        it as an improvement. The 'path' parameter specifies the file path where the best model will be saved when
        the early stopping condition is met.
        """
        early_stopping = EarlyStopping(patience=5, delta=0.1, path=model_file)

        # Instantiate the optimizer. Here, the Adam optimizer is used. The optimizer takes the model's learnable
        # parameters and the defined learning rate as inputs and is responsible for updating the model's parameters
        # during the training process to minimize the loss function.
        optimizer = optim.Adam(params=mymodel.parameters(), lr=lr)
        """
        Instantiate the learning rate scheduler. Here, StepLR is used, which adjusts the learning rate according to
        the specified step size and gamma factor. Every'step_size' epochs, the learning rate will be multiplied by
        'gamma' for decay. The 'last_epoch' parameter is set to -1, indicating that the initial epoch number is -1,
        and it will be updated based on the actual training progress later.
        """
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=-1)
        """
        Call the 'train' function to train the model. The function takes the model object, the number of training epochs (51),
        the optimizer, the learning rate scheduler, the loss function, the device, the DataLoader for the training set,
        the DataLoader for the test set, and the early stopping object as parameters. During the training process,
        multiple epochs of training will be performed to update the model's parameters, and the early stopping mechanism
        will be used to determine whether to stop the training early based on certain conditions.
        """
        train(mymodel, 51, optimizer, scheduler, criterion, device, trainloader, testloader, early_stopping)

        # Load the trained model from the saved model file and move it to the specified device for subsequent prediction
        # and other operations.
        load_model = torch.load(model_file).to(device)

        """
        Call the 'Predict' function to make predictions and obtain relevant evaluation metrics (such as the R-squared
        value, a correlation coefficient 'p', etc.), as well as the true label values and the predicted values. The
        function takes the device, the loaded model, the loss function, and the DataLoader for the test set as inputs
        and is used to make predictions on the test set and calculate the evaluation metrics.
        """
        r2, p, y, out, _ = Predict(device, load_model, criterion, testloader)

        # Concatenate the true label values obtained in this iteration to the overall array of true label values 'y_all'.
        y_all = np.concatenate((y, y_all))
        # Concatenate the predicted values obtained in this iteration to the overall array of predicted values 'out_all'.
        out_all = np.concatenate((out, out_all))

        # Append the R-squared value obtained in this iteration to the list of all R-squared values 'all_r'.
        all_r.append(r2)
        # Append the correlation coefficient obtained in this iteration to the list of all correlation coefficients 'all_pcc'.
        all_pcc.append(p)

    # Append the average of all the R-squared values in the 'all_r' list to the list itself.
    all_r.append(np.average(all_r))
    # Append the average of all the correlation coefficients in the 'all_pcc' list to the list itself.
    all_pcc.append(np.average(all_pcc))
    print(all_pcc)
    # Create a pandas DataFrame with columns 'Y_TRUE' and 'Y_Predict', using the concatenated arrays of true label values
    # and predicted values respectively.
    df = pd.DataFrame({'Y_TRUE': y_all, 'Y_Predict': out_all})
    # Save the DataFrame to a text file. The file path includes the phenotype name and the model name, and the 'index'
    # parameter is set to False to avoid saving the row index in the file.
    df.to_csv(f'{phenos}_result_{modelname}_Predict.txt', index=False)
    # Create another pandas DataFrame with columns 'R2' and 'PCC', using the lists of R-squared values and correlation
    # coefficients respectively.
    df = pd.DataFrame({'R2': all_r, 'PCC': all_pcc})
    # Save this DataFrame to a text file with a tab delimiter. Again, the 'index' parameter is set to False to avoid
    # saving the row index.
    df.to_csv(f'{phenos}_result_{modelname}.txt', sep='\t', index=False)
