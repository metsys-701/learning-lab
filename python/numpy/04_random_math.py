"""
Topic: NumPy Random and Math Functions

Description:
This file contains my practice exercises for NumPy random number generation and math functions.
"""
import numpy as np

#=========================
#1. Random Integer Arrays
#=========================
# Generate random integers.
arr1 = np.random.randint(1, 100, 10)
print("Random integers:", arr1)

#=========================
#2. Random Float Arrays
#=========================
# Generate random floating-point numbers.
arr2 = np.random.rand(5)
print("Random floats:", arr2)

#=========================
#3. Matrix of Random Numbers
#=========================
# Generate a random matrix.
arr3 = np.random.randint(1,50, size=(3,3))
print("Random matrix:\n", arr3)

#=========================
#4. Basic Math Functions
#=========================

arr4 = np.array([12, 85, 43, 19, 91, 7, 54])

print("Maximum value:",arr4.max())
print("Minimum value:",arr4.min())
max_index = np.argmax(arr4)
print("Index of maximum value:",max_index)
min_index = np.argmin(arr4)
print("Index of minimum value:",min_index) 

#=========================
#5. Sorting Arrays
#=========================
# Sort a NumPy array.

sorted_arr = np.sort(arr4)
print("Sorted array:", sorted_arr)

#=========================
#6. Unique Values
#=========================
# Find unique values.

arr5 = np.array([5, 8, 5, 2, 8, 9, 2, 1, 5])
unique_values = np.unique(arr5)
print("Unique values:", unique_values)