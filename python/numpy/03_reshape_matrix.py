"""
Topic: NumPy Reshape and Matrices

Description:
This file contains my practice exercises for reshaping arrays and working with matrices.
"""

import numpy as np

#=========================
#1. Creating a Matrix
#=========================
# Reshape an array into a matrix.
arr1= np.array([1, 2, 3, 4, 5, 6])
arr2 = arr1.reshape(2,3)
print(arr2)


#=========================
#2. Matrix Information
#=========================
# Print matrix information.

print(arr2.shape)
print(arr2.ndim)
print(arr2.dtype)
print(arr2.size)

#=========================
#3. Accessing Matrix Elements
#=========================
# Access matrix rows and columns.
arr2 = arr1.reshape(2,3)
print("first row:",arr2[0])
print("Second row:", arr2[1])
print("first column",arr2[:,0])
print("second column", arr2[:, 1])
print(arr2[1,2])

#=========================
#4. Matrix Slicing
#=========================

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("first two row:",matrix[0:2, :])
print("last two row:",matrix[1:, :])
print("first two columns:", matrix[:, 0:2])
print("last two columns:", matrix[:, 1:])


