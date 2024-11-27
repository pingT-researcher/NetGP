from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# Read label data from 'phenos_match.txt', set header at first row and tab as delimiter.
# Then drop the 'ID' column, transpose the remaining data and print its shape for inspection.
labels = pd.read_csv('phenos_match.txt', header=0, sep='\t')
labels = np.array(labels.drop(columns=['ID'])).T
print(labels.shape)

# Loop for likely different label subsets or scenarios (indexed by 0 and 1 here).
for i in [0, 1]:
    # Initialize empty arrays to store correlation coefficients and p-values respectively during the loop.
    cors = np.array([])
    ps = np.array([])

    # Read data from 'hz.012.10W_match.txt', handle low memory setting and fill missing values with 0. Then drop specific
    # columns not needed for core analysis and transpose the data to a format suitable for later correlation
    # calculations.
    deinfo = pd.read_csv('hz.012.10W_match.txt', sep='\t', low_memory=False).fillna(0)
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    datas = np.array(deinfo.drop(columns=columns_to_skip), dtype=int).T
    print(datas.shape)
    print(labels.shape)

    # Iterate over each row of data (representing data for each locus).
    for data in datas:
        # Calculate Pearson correlation coefficient and corresponding p-value between current locus data and specific
        # label subset. If the result contains NaN (invalid values), set correlation coefficient to 0.
        cor, p = pearsonr(data, labels[i])
        if np.isnan(cor + p):
            cor = 0
        print(cor, p)

        # Append absolute value of correlation coefficient and p-value to respective arrays for later analysis.
        cors = np.append(cors, np.abs(cor))
        ps = np.append(ps, p)

    # Print the average of the correlation coefficients to get an idea of overall correlation strength.
    print(np.average(cors))
    # Find indices where the correlation is both statistically significant (p-value < 0.05) and stronger than the
    # average correlation.
    result = np.where((ps < 0.05) & (cors > np.average(cors)))[0]

    # Filter correlation coefficients based on the obtained indices.
    filtered_cors = cors[result]
    # Get indices that would sort the filtered correlation coefficients in descending order.
    sorted_indices = np.argsort(filtered_cors)[::-1]
    # Use these sorted indices to get the corresponding original indices in the correct order.
    sorted_result_indices = result[sorted_indices]
    # Save these sorted result indices to a text file for later use.
    np.savetxt('HeadingDataFeatures.txt', sorted_result_indices)

    # Update the result variable and perform some index concatenation operation (exact purpose depends on context).
    result = sorted_result_indices
    result = np.concatenate((np.array([0, 1, 2, 3, 4, 5]), result + 6))
    print(sorted_result_indices)
    print(sorted_result_indices.shape)

    # Transpose the data frame for easier row-based indexing and filtering.
    deinfo = deinfo.T
    # Filter the data frame based on the obtained indices to keep relevant rows.
    deinfo = deinfo.iloc[result].T
    print(deinfo)
    # Save the filtered data frame to a text file without row indices.
    deinfo.to_csv(f'PHeadingData.txt', index=False)
