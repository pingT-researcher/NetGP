import numpy as np
import pandas as pd

# Read data from file, handle missing values and get columns in proper format
deinfos = pd.read_csv('hz.012.10W_match.txt', sep=',', low_memory=False, header=0).fillna(0)
columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
datas = np.array(deinfos.drop(columns=columns_to_skip), dtype=int).T

# Read and process feature indices from another file
feature_index = np.array(pd.read_csv('HeadingDataFeatures.txt', header=None), dtype=int).flatten()
datas = datas[feature_index]
print(datas.shape)

num = len(datas)
index = np.array([i for i in range(num)])
print(index.shape)

# Initialize set to store indices of data to remove
to_remove = set()
# Check for rows with homogeneous data and mark them for removal
for i in range(len(datas)):
    if np.all(datas[i] == datas[i][0]):
        to_remove.add(i)

# Calculate correlation matrix for data
ld_matrix = np.corrcoef(datas)
# Analyze correlation matrix and mark more indices for removal based on conditions
for i in range(len(datas)):
    if i in to_remove:
        continue
    for j in range(i + 1, len(datas)):
        if not np.isnan(ld_matrix[i][j]) and np.abs(ld_matrix[i][j]) < 0.7:
            continue
        else:
            to_remove.add(j)

# Convert set of indices to remove to numpy array
to_remove = np.array(list(to_remove), dtype=int)
print(to_remove.shape)
# Get remaining valid indices
result = index[~np.isin(index, to_remove)]
print(result.shape)
# Update feature indices file with remaining valid ones
np.savetxt('HeadingDataFeatures.txt', feature_index[result])

# Concatenate indices (for some specific purpose related to data handling)
result = np.concatenate((np.array([0, 1, 2, 3, 4, 5]), result + 6))
# Transpose data and filter based on valid indices
deinfo = deinfos.T
deinfo = deinfo.iloc[result].T
# Save filtered data to new file
deinfo.to_csv('PHeadingData.txt', index=False)