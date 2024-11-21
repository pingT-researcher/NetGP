import numpy as np
import pandas as pd


# Define a function named 'Datasets' which is used to load and process datasets as well as corresponding label data
# based on different input parameters.
def Datasets(type, phnoe, labelpath):
    global datas, labels
    # Check the value of the 'type' parameter to determine which type of data to load.
    if type == 'snp':
        # Read the file named 'P{phnoe}.txt' (comma-separated) as the data. Here, 'phnoe' is supposed to be a
        # variable representing a specific identifier.
        datas = pd.read_csv(f'P{phnoe}.txt', delimiter=',')
        # Define a list of column names to be skipped. The data in these columns might not be needed in the
        # subsequent processing.
        columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
        # Remove the specified columns from the read data and convert it to a NumPy array. Meanwhile, specify the
        # data type as integer.
        datas = np.array(datas.drop(columns=columns_to_skip), dtype=int)

    elif type == 'trans':
        # Read the file named 'gene_trans_{phnoe}.txt' (tab-separated) as the data. Use a lambda function to filter
        # columns (exclude the column named 'Geneid'). Set the header to be in the first row and the data type to be
        # double precision floating point numbers. Finally, perform a transpose operation.
        datas = np.array(pd.read_csv(
            f'gene_trans_{phnoe}.txt', sep='\t',
            usecols=lambda col: col != 'Geneid', header=0, dtype=np.double)).T
    elif type == 'pca':
        # Read the file named 'hz.012.10W_data.tsv' (tab-separated with the header in the first row) as the data.
        # This might be data related to Principal Component Analysis (PCA).
        datas = pd.read_csv(f'hz.012.10W_data.tsv', header=0, delimiter='\t')
    # Read the label data. The file path is specified by the 'labelpath' parameter and is tab-separated.
    labels = pd.read_csv(labelpath, sep='\t')
    # Define the column name to be skipped (here it's the 'id_name' column). The data in this column might not be
    # needed in the subsequent processing.
    columns_to_skip = ['id_name']
    # Remove the specified column from the label data and convert it to a NumPy array.
    labels = np.array(labels.drop(columns=columns_to_skip))
    # Return the processed dataset and the label data (the label data is also converted to the NumPy array format).
    return datas, np.array(labels)
