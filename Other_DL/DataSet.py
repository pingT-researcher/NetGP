import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import scipy.sparse as sp


# Custom dataset class, inheriting from torch.utils.data.Dataset, used to create dataset objects that meet the requirements of PyTorch
class CustomDataset(data_utils.Dataset):
    def __init__(self, data, labels):
        """
        Initialization function that receives data and corresponding labels and saves them as class attributes.
        These data and labels will be used later in operations like data loading.
        :param data: The input data, such as feature data, etc., which can be in various formats (like numpy arrays, etc.) and will be converted to tensors later.
        :param labels: The corresponding data labels, which have a one-to-one correspondence with the data samples and will also be converted to tensors later.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Returns the length of the dataset, that is, the number of data samples. This method is required for implementing the Dataset abstract class.
        :return: The length of the dataset, i.e., the number of data samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves the corresponding data sample and label according to the given index. This method is also required for implementing the Dataset abstract class.
        When using DataLoader to load data, this method will be used to obtain samples from the dataset one by one.
        :param index: The index of the sample to be retrieved.
        :return: Returns the sample corresponding to the index and its corresponding label.
        """
        sample = self.data[index]
        label = self.labels[index]
        return sample, label


def divide_Dataset(datas, labels, K_num, fold_num, batch_size, modeltype=''):
    """
    Function for splitting the dataset. Based on different conditions (such as whether a specific model type is
    specified, the fold number in cross-validation, etc.), it splits the input data into training and test sets and
    wraps them into PyTorch's DataLoader objects respectively for subsequent operations like batch loading during
    training. :param datas: The input dataset, which can be in various formats (like numpy arrays, etc.),
    usually feature data. :param labels: The label data corresponding to the dataset, which is consistent with the
    order of the dataset samples. :param K_num: A parameter used to determine how to split the dataset. For example,
    in a cross-validation scenario, it represents the current fold number, etc. When it is 0, it may indicate a
    special splitting method (like using all data as both the training and test sets). :param fold_num: A parameter
    related to the number of samples in each fold, used to calculate the index range for splitting the dataset.
    Combined with K_num, it determines the ranges of the training and test sets. :param batch_size: Used to set the
    size of each batch in the DataLoader, controlling the amount of data passed into the model in each iteration.
    :param modeltype: A string representing the model type, used to perform specific data processing according to
    different models. For example, for the 'DNNGP' model, there will be special processing. If it is an empty string,
    no special processing related to the model will be performed. :return: Returns the DataLoader for the training
    set and the DataLoader for the test set, which are convenient for later use in training and testing the model.
    """
    if modeltype == 'DNNGP':
        num = len(datas[0])
        size = int(np.ceil(np.sqrt(np.ceil(num / 24))))
        arr = np.zeros((len(datas), size * size * 24 - num))
        # Concatenate the original data and the zero-filled array along the column axis
        datas = np.concatenate((datas, arr), axis=1)
        datas = datas.reshape(len(datas), 24, size, size)
    if K_num == 0:
        train_data = torch.tensor(datas)
        train_label = torch.tensor(labels)
        test_data = torch.tensor(datas)
        test_label = torch.tensor(labels)
    else:
        start_index = (K_num - 1) * fold_num
        end_index = K_num * fold_num
        print(start_index, end_index)
        train_data = torch.tensor(np.concatenate((datas[0:start_index], datas[end_index:])))
        train_label = torch.tensor(np.concatenate((labels[0:start_index], labels[end_index:])))

        test_data = torch.tensor(datas[start_index:end_index])
        test_label = torch.tensor(labels[start_index:end_index])

    trainset = CustomDataset(train_data, train_label)
    trainloader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = CustomDataset(test_data, test_label)
    testloader = data_utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def Datasets(types, datas_file, label_file, gene_re=''):
    """
    Reads data and labels from corresponding files according to the specified dataset type (such as 'pca','snp',
    etc.) and performs some specific data processing (such as filtering data based on gene-related files,
    etc.). :param types: A string representing the dataset type, used to determine from which format of files to read
    data and how to perform subsequent processing. For example, different types like 'pca' and'snp' have different
    reading and processing logics. :param datas_file: The path to the data file, pointing to the file that stores the
    dataset. The format varies depending on the type (such as.tsv,.csv, etc.). :param label_file: The path to the
    label file, pointing to the file that stores the corresponding dataset labels, usually in.tsv format. :param
    gene_re: The path to the gene-related file (an optional parameter). :return: Returns the processed dataset (
    usually in numpy array format) and labels (usually in numpy array format) according to different situations. If
    it is'snp' type and the gene_re parameter has a value, it will also return the tensor corresponding to the gene
    index (of torch.tensor type).
    """
    if types == 'pca':
        datas = pd.read_csv(f'hz.012.10W_data.tsv', header=0, delimiter='\t')
        datas = np.array(datas.drop(columns=['ID']))
    if types == 'snp':
        datas = pd.read_csv(datas_file, sep=',', low_memory=False).fillna(0)
        columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
        datas = np.array(datas.drop(columns=columns_to_skip), dtype=np.double)
    else:
        datas = np.array(
            pd.read_csv(datas_file, sep='\t', usecols=lambda col: col != 'Geneid', skiprows=0, dtype=np.double)).T

    labels = pd.read_csv(label_file, sep='\t')
    columns_to_skip = ['id_name']
    labels = np.array(labels.drop(columns=columns_to_skip))

    if gene_re != '' and types == 'snp':
        datas = datas.T
        with open(gene_re, 'r') as gg:
            genes = [[int(x) for x in line.split()[1:]] for line in gg.readlines()]
            genes_index = np.full((len(genes), len(datas)), 0)
            for row_idx, row in enumerate(genes):
                genes_index[row_idx, row] = 1

        genes_index = genes_index.T
        rows_to_keep = np.any(genes_index, axis=1)
        genes_index = genes_index[rows_to_keep]
        # Remove SNP features that have no gene connection
        datas = datas[rows_to_keep].T
        return np.array(datas), np.array(labels), torch.tensor(np.array(genes_index))

    return np.array(datas), np.array(labels)
