import numpy as np
import pandas as pd

# Read the data from the file, tab-separated
trans_info = pd.read_csv('all_Rpkm_samplename_match.txt', sep='\t')

# Get the labels from the first row (column names except the first one)
first_row_labels = trans_info.columns.values[1:]
print(first_row_labels)

# Extract the first column (gene IDs)
genID_column = list(trans_info.iloc[:, 0])

# Get the data part (excluding the first column) as a numpy array
data = np.array(trans_info.iloc[:, 1:])

# Create a list of zeros with the same length as the number of columns in data
no_gene = [0 for i in range(len(data[0]))]

locs = []
# Read the gene location file, split each line and get the first element (location)
with open('snp_gene_HeadingDate.txt', 'r') as ss:
    for s in ss.readlines():
        locs.append(s.split()[0])
print(locs)

sub_gene = []
# For each location, find the corresponding gene data or add zeros if not found
for loc in locs:
    if loc in genID_column:
        num = genID_column.index(loc)
        sub_gene.append(data[num])
    else:
        sub_gene.append(no_gene)

# Create a DataFrame with the subsetted gene data, using locs as index and first_row_labels as columns
df = pd.DataFrame(sub_gene, index=locs, columns=first_row_labels)

# Save the DataFrame to a CSV file, tab-separated
df.to_csv('gene_trans_HeadingDate.txt', sep='\t')