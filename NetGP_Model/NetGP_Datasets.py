import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils


# Assume that data_utils.Dataset inherits from a base dataset class (such as the Dataset class in PyTorch) and is
# used for creating custom datasets.
class CustomDataset(data_utils.Dataset):
    def __init__(self, data, labels, transcribe):
        """
        Initialization function for creating an instance of the CustomDataset class. It takes data, labels,
        and transcription-related data as parameters and saves them as instance attributes.

        Parameters:
        - data: The input data(SNP),
        - labels: The corresponding data labels
        - transcribe: Transcription-related data
        """
        self.data = data
        self.labels = labels
        self.transcribe = transcribe

    def __len__(self):
        """
        This special method is used to return the length of the dataset, that is, the number of samples in the dataset.

        Returns: - Returns the number of samples contained in the dataset. It is determined by getting the length of
        self.data, because self.data stores all the sample data, and its length represents the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        This special method is used to obtain a single sample from the dataset according to the given index,
        enabling the dataset object to be accessed like a list through the index.

        Parameters: - index: The index position of the sample to be obtained in the dataset. The value range should
        be between 0 and the length of the dataset minus 1 (that is, 0 <= index < len(self.data)).

        Returns:
        - sample: The data sample at the specified index position in the dataset.
        - label: The label corresponding to the data sample.
        - trans: The transcription-related data corresponding to the data sample.
        """
        sample = self.data[index]
        label = self.labels[index]
        trans = self.transcribe[index]
        return sample, label, trans


def Datasets(snp_file, label_file, gene_re, gNet_re, transcribe_file):
    """
    Load and preprocess genetic data files and return relevant data.

    Args:
    - snp_file: SNP data file path.
    - label_file: Label data file path.
    - gene_re: Gene relationship file path.
    - gNet_re: Gene network relationship file path.
    - transcribe_file: Transcription data file path

    Returns:
    - datas: Processed SNP data as NumPy array.
    - labels: Processed label data as NumPy array.
    - genes_index: Processed gene relationship as PyTorch tensor.
    - sparse_matrix: Gene network relationship as PyTorch tensor.
    - transcribe: Processed transcription data as NumPy array.
    """
    # Read SNP data, transpose it, and print its shape
    datas = pd.read_csv(snp_file, sep=',', low_memory=False).fillna(0)  # 样本数 X 位点数
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    datas = np.array(datas.drop(columns=columns_to_skip), dtype=np.double)
    # Read label data, skip some columns, and convert to NumPy array. Then print its shape
    labels = pd.read_csv(label_file, sep='\t')
    columns_to_skip = ['id_name']
    labels = np.array(labels.drop(columns=columns_to_skip))
    # Process gene relationship data to create genes_index tensor
    with open(gene_re, 'r') as gg:
        genes = [[int(x) for x in line.split()[1:]] for line in gg.readlines()]
        genes_index = np.full((len(genes), len(datas[0])), 0, dtype=np.float)
        for row_idx, row in enumerate(genes):
            genes_index[row_idx, row] = 1
    # Read gene network relationship data and create sparse_matrix tensor

    node_relationship = np.array(pd.read_csv(gNet_re, sep='\t', header=None, dtype=np.float))

    sparse_matrix = np.full((len(genes), len(genes)), 0, dtype=np.double)
    for node in node_relationship:
        i, j, v = node
        sparse_matrix[int(i), int(j)] = v
        sparse_matrix[int(j), int(i)] = v
    print("sparse_matrix处理前:", np.count_nonzero(sparse_matrix))
    # Do some processing on sparse_matrix
    for i in range(len(sparse_matrix)):
        neighbors = np.nonzero(sparse_matrix[i] > 1)[0]
        for j in neighbors:
            jneighbors = np.nonzero(sparse_matrix[j] > 1)[0]
            sparse_matrix[i][jneighbors] = np.maximum(sparse_matrix[i][jneighbors], 1)
            sparse_matrix[i][i] = 0
    print("sparse_matrix处理后:", np.count_nonzero(sparse_matrix))
    # Process transcription data, generate random if file is empty
    if transcribe_file == "":
        transcribe = np.random.rand(len(labels[0]), len(gene_re))
    else:
        transcribe = np.array(pd.read_csv(transcribe_file, sep='\t', usecols=lambda col: col != 'Geneid', skiprows=0,
                                          dtype=np.double)).T
    return np.array(datas), np.array(labels), torch.tensor(genes_index), torch.tensor(sparse_matrix), transcribe


def divide_Dataset(datas, labels, transcribe, K_num, num, batch_size):
    """
    This function is used to divide the input datasets (including data, labels, and transcription data) into training and testing sets.
    It also creates data loaders for both the training and testing sets, which can be used to feed data to a model during the training and evaluation processes.

    Parameters:
    - datas: The input data, usually representing features of samples. It's typically in a format like a NumPy array or a tensor.
    - labels: The corresponding labels for the input data, indicating the target values or classes for each sample.
    - transcribe: Transcription-related data, which might be related to gene transcription or other relevant information depending on the context.
    - K_num: An indicator to determine how to split the data. If K_num is 0, it might mean using the whole dataset for both training and testing (or some special handling).
              If K_num is other values, it indicates a specific fold for cross-validation or another splitting strategy.
    - num: The number of samples in each fold or a specific size parameter related to the splitting. Its meaning depends on the overall splitting logic.
    - batch_size: The number of samples in each batch when creating the data loaders. It determines how many samples are fed to the model at once during training or testing.

    Returns:
    - trainloader: A data loader for the training set, which can be used to iterate over the training data in batches.
    - testloader: A data loader for the testing set, which can be used to iterate over the testing data in batches.
    """
    if K_num == 0:
        """
        If K_num is 0, this branch is executed. In this case, it seems to use the entire dataset for both training and testing.
        This might be useful for scenarios like initial testing or when not performing cross-validation.
        """
        train_data = torch.tensor(datas)
        train_label = torch.tensor(labels)
        train_trans = torch.tensor(transcribe)
        test_data = torch.tensor(datas)
        test_label = torch.tensor(labels)
        test_trans = torch.tensor(transcribe)
    else:
        """If K_num is not 0, this branch is executed to perform a specific splitting of the dataset based on the 
        given indices. It's likely related to cross-validation or a similar splitting strategy where different parts 
        of the dataset are used for training and testing."""
        start_index = (K_num - 1) * num
        end_index = K_num * num
        print(start_index, end_index)
        train_data = torch.tensor(np.concatenate((datas[0:start_index], datas[end_index:])))
        train_label = torch.tensor(np.concatenate((labels[0:start_index], labels[end_index:])))
        train_trans = torch.tensor(np.concatenate((transcribe[0:start_index], transcribe[end_index:])))

        test_data = torch.tensor(datas[start_index:end_index])
        test_label = torch.tensor(labels[start_index:end_index])
        test_trans = torch.tensor(transcribe[start_index:end_index])

    trainset = CustomDataset(train_data, train_label, train_trans)
    """Create a CustomDataset instance for the training set. This custom dataset combines the training data, labels, 
    and transcription data. It will be used to create the data loader for the training set."""
    trainloader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    """Create a data loader for the training set. It will load the training data in batches of size batch_size. 
    Shuffling the data is enabled (shuffle=True) to randomize the order of samples in each epoch during training. 
    num_workers=3 indicates that 1 subprocesses will be used to load the data in parallel, which can speed up the 
    data loading process."""

    testset = CustomDataset(test_data, test_label, test_trans)
    """
    Create a CustomDataset instance for the testing set in a similar way as for the training set.
    """
    testloader = data_utils.DataLoader(testset, batch_size=300, shuffle=False, num_workers=1)
    """Create a data loader for the testing set. The batch size is set to 300, and shuffling is disabled (
    shuffle=False) as the order of the test data usually doesn't need to be randomized. Again, num_workers=1 is used 
    for parallel data loading."""

    return trainloader, testloader
