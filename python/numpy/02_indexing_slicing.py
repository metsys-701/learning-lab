"""
Topic: NumPy Indexing and Slicing

Description:
This file contains my practice exercises for NumPy indexing and slicing.
"""

import numpy as np

#=========================
#1. Basic Indexing
#=========================
# Access array elements using indexes.
arr1 = np.array([10, 20, 30, 40, 50])

print(arr1[0])
print(arr1[2])
print(arr1[4])

#=========================
#2. Negative Indexing
#=========================
# Access array elements using negative indexes.
arr2 = np.array([100, 200, 300, 400, 500])

print(arr2[-1])
print(arr2[-2])
print(arr2[-3])

#=========================
#3. Basic Slicing
#=========================
# Slice a NumPy array.
arr3 = np.arange(1,10)
print(arr3[:3])
print(arr3[-3:])
print(arr3[3:])
print(arr3[:5])

#=========================
#4. Step Slicing
#=========================
# Slice an array using step values.

print(arr3[::2])
print(arr3[::3])
print(arr3[::-1])

#=========================
#5. Combining Indexing and Slicing
#=========================
# Combine indexing and slicing.
arr4 = np.array([5, 10, 15, 20, 25, 30, 35, 40])

print(arr4[:4]) #result [5, 10, 15, 20]
print(arr4[-4:]) #result [25, 30, 35, 40]
print(arr4[::2]) # result [5, 15, 25, 35]
print(arr4[1::2]) # result [10, 20, 30, 40]