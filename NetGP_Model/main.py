import numpy as np
import pandas as pd
import torch
from torch import optim
from NetGP_Datasets import Datasets, divide_Dataset
from predict import Predict
from train import train
# from NetGP import NetGP
from NetGP_Snp import NetGP
# from NetGP_Trans import NetGP
from EarlyStopping import EarlyStopping

# Set the device to 'cuda:0' if GPU is available, otherwise 'cpu'.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the default tensor type to DoubleTensor.
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    # HeadingDate lr=0.3 bach_size=32 step_size = 20
    # FT10 lr = 0.2  bach_size=32  step_size = 25

    
    # Phenotype name, which might represent a certain biological feature to predict.
    phenos = 'HeadingDate'
    # Name of the model.
    modelname = 'netgp'
    # Information about the dataset, like its origin.
    data_info = '华中农大数据集'
    # File path of SNP data.
    snp_file = f'HeadingDate.txt'
    # File path of label data.
    label_file = f'phenos_match.txt'
    # File path related to gene data.
    gene_re = f'snp_gene_HeadingDate.txt'
    # Another file path related to gene network.
    gNet_re = f'gene_re_ricenet_HeadingDate.txt'
    # File path of transcription data.
    transcribe_file = f'gene_trans_HeadingDate.txt'

    # Load datasets using the Datasets function, get data, labels, gene indices and transcription data.
    datas, labels, genes_index, gene_re_index, transcribe = Datasets(snp_file, label_file, gene_re, gNet_re,
                                                                     transcribe_file)
    # Print the shapes of data, labels and other related variables for checking.
    print('data:', datas.shape)
    print('labels', labels.shape)
    print('genes_index:', genes_index.shape)
    print(gene_re_index.shape)
    print(transcribe.shape)
    # Get the first label value and print it.
    label = labels[0]
    # File path of random index file.
    random_index_file = f'random_index_HeadingDate.txt'
    # Read the random index file and get a flattened array for shuffling data.
    random_permutation = np.array(pd.read_csv(random_index_file, sep='\t')).flatten()
    # Shuffle data, transcription data and label based on the random index.
    data = datas[random_permutation]
    transcribe = transcribe[random_permutation]
    label = label[random_permutation]

    # Define the loss function as SmoothL1Loss.
    criterion = torch.nn.SmoothL1Loss()

    all_r = []
    all_pcc = []
    y_all = np.array([])
    out_all = np.array([])

    for num in [1]:
    # Finding the right initial hyperparameters
      # for  bach_size in [64,40,32,24,20,16]: 
      # for step_size in [20,25,30,35,40]:  
       # for lr in [0.3,0.2,0.1,0.01]:  
        
        # Set learning rate.
        lr = 0.3
        # Set batch size.
        bach_size = 32
        # Set step size for learning rate scheduler.
        step_size = 30
        # Divide dataset into train and test sets, also get average label.
        trainloader, testloader = divide_Dataset(data, label, transcribe, num, 28, bach_size)
        # Create a NetGS model instance and move it to the device.
        mymodel = NetGP(genes_index, gene_re_index).to(device)
        # Set the model file path for saving and loading.
        model_file = f'{phenos}_{num}_{modelname}.model'
        # Create an EarlyStopping instance to control early stopping during training.
        early_stopping = EarlyStopping(patience=10, delta=0.01, path=model_file)
        # Create an Adamax optimizer for model parameter update.
        optimizer = optim.Adamax(params=mymodel.parameters(), lr=lr)
        # Create a learning rate scheduler.
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=-1)
        # Train the model with given parameters.
        train(mymodel, 150, optimizer, scheduler, criterion, device, trainloader, testloader, early_stopping)
        # Load the saved model.
        load_model = torch.load(model_file).to(device)
        # Use the loaded model to predict and get evaluation metrics and prediction results.
        r2, p, y, out, _ = Predict(device, load_model, criterion, testloader)
        # Concatenate the true labels and prediction results for later analysis.
        y_all = np.concatenate((y, y_all))
        out_all = np.concatenate((out, out_all))
        # Save the R2 score and correlation coefficient.
        all_r.append(r2)
        all_pcc.append(p)
    # Add the average R2 score and correlation coefficient to the lists.
    all_r.append(np.average(all_r))
    all_pcc.append(np.average(all_pcc))
    print(all_pcc)
    # Create a DataFrame for true labels and prediction results and save it as a CSV file.
    df = pd.DataFrame({'Y_TRUE': y_all, 'Y_Predict': out_all})
    df.to_csv(f'{phenos}_result_{modelname}_Predict.txt', index=False)
    # Create a DataFrame for R2 score and correlation coefficient and save it as a CSV file.
    df = pd.DataFrame({'R2': all_r, 'PCC': all_pcc})
    df.to_csv(f'{phenos}_result_{modelname}.txt', sep='\t', index=False)
