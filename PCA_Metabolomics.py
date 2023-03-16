# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:49:44 2023

@author: crtuser
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Load data
Liver = pd.read_excel('C:\\Users\\crtuser\\OneDrive - TCDUD.onmicrosoft.com\\Documents\\PhD\\Project\\Data\\MiceWeightLossCohort\\MOUSE_Metabolomics\\Evanna raw file.xlsx', sheet_name = 'Liver')

# Filter data
ml_df_train = Liver[Liver['Group'].isin(['HFD8', 'WL', 'SFD', 'HFD1'])]

# Prepare X matrix
X = ml_df_train.drop(['Group', 'Diet', 'Index', 'Tissue'], axis=1)

# Perform PCA
n_components = 5 # number of components to extract
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = [f"PC{i+1}" for i in range(n_components)])

# Concatenate with y and plot
y = ml_df_train['Group']
finalDf = pd.concat([principalDf, y], axis=1)

plt.figure(figsize=(10, 10))
ax = sns.scatterplot('PC1', 'PC2', data=finalDf, hue='Group', s = 200, palette = 'colorblind')
plt.suptitle('', fontsize=18)
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
plt.legend(fontsize='xx-large', title_fontsize='xx-large')
plt.show()

# Get loadings
x = pd.DataFrame(pca.components_[:n_components,:],columns=X.columns,index = [f"principal component {i+1}" for i in range(n_components)]).T

# Sort variables by correlation to each principal component and store in list
pcs = []
for i in range(n_components):
    pc_name = f"principal component {i+1}"
    pc = x.sort_values(pc_name, ascending=False)[pc_name].index.tolist()[:5]
    pcs.append(pc)

# Write sorted variables to CSV file
with open('Liver_PCA.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, pc in enumerate(pcs):
        writer.writerow([f"Principal component {i+1}"] + pc)
