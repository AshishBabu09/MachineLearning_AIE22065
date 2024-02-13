import pandas as pd
import numpy as np
 
# Load data from Excel file
file_path = "D:\python\Machine_Learning\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")
 
# Segregate data into matrices A and C
A = df.iloc[:, 1:4].values  # Assuming the relevant columns are 2nd, 3rd, and 4th
C = df.iloc[:, 4].values  # Assuming the 5th column is the payment column
 
# Dimensionality of the vector space
dimensionality = A.shape[1]
 
# Number of vectors in the vector space
num_vectors = A.shape[0]
 
# Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)
 
# Using Pseudo-Inverse to find the cost of each product
pseudo_inverse_A = np.linalg.pinv(A)
cost_vector = np.dot(pseudo_inverse_A, C)
 
# Display results
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product:", cost_vector)