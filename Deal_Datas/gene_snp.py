import numpy as np
import pandas as pd

# Read data from file, convert to numpy array and flatten
index = np.array(pd.read_csv('HeadingDateFeature.txt', sep='\n', header=None), dtype=int).ravel()

# Open two files, one for reading data and one for writing processed data
with open('snp_gene_HeadingDate.txt', 'r') as w, open('snp_gene.txt', 'r') as f:
    # Loop through each line in the input file
    for line in f.readlines():
        row_str = ''
        # Process line data, adjust indices and find intersection with 'index'
        dat = [int(x) - 1 for x in line.split()[1:]]
        dat = list(set(dat) & set(index))
        # If there's relevant data, start building row string
        if len(dat) > 0:
            row_str += line.split()[0] + '\t'
            # Add indices of relevant data to row string
            row_str += '\t'.join([str(np.where(index == x)[0][0]) for x in dat])
            w.write(row_str + '\n')