"""
Topic: NumPy Broadcasting

Description:
This file contains my practice exercises for NumPy broadcasting.
"""
import numpy as np

#=========================
#1. Array + Number
#=========================
# Add a scalar value to an array.
arr = np.array([5, 10, 15, 20, 25])

new_arr = arr + 10
print(new_arr)

#=========================
#2. Matrix + Number
#=========================
# Multiply a matrix by a scalar value.
arr1 = np.array([[1, 2], [3, 4]])
print(arr1)
new_matrix = arr1 *5
print(new_matrix)

#=========================
#3. Array + Array
#=========================
# Add two NumPy arrays.

first_array = np.array([1, 2, 3])
second_array = np.array([10, 20, 30])
sum_array = first_array + second_array
print(sum_array)

#=========================
#4. Matrix + Array
#=========================
# Apply broadcasting between a matrix and an array.

matrix = np.array([[1, 2, 3],[4, 5, 6]])
array = np.array([10, 20, 30])

broadcast_result = matrix + array
print(broadcast_result)


#=========================
#5. Broadcasting Challenge
#=========================

matrix_3x3 = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(matrix_3x3)
matrix_plus_100 = matrix_3x3 + 100
print(matrix_plus_100)
matrix_divided_by_2 = matrix_3x3 /2
print(matrix_divided_by_2)