# Import the NumPy library.
import numpy as np 


# =========================
# 1. Creating an array
# =========================

arr = np.array([10, 20, 30, 40, 50])
print("Array:",arr)
print("Data type:",arr.dtype)
print("Dimensions:",arr.ndim)
print("Shape:",arr.shape)    
print("Number of elements:",arr.size)

# =========================
# 2. Accessing an element
# =========================


arr2= np.array([5 ,10, 15, 20, 25])
print(arr2[2])

#=========================
#3. Array Arithmetic Operations
#=========================

arr3 = np.array([2, 4, 6, 8, 10])
print(arr3)
print(arr3 * 2)


#=========================
#4. Boolean Indexing
#=========================

arr4 = np.array([12, 18, 5, 42, 7, 30])
filtered_array=[]
print(arr4 >20)
filtered_array = arr4[arr4 >20]
print(filtered_array)

#=========================
#5. Basic Statistics
#=========================

arr5 = np.array([15, 25, 35, 45, 55])
print("Maximum value:",arr5.max())
print("Minimum value:",arr5.min())
print("Sum:",arr5.sum())
print("Average:", arr5.mean())

#=========================
#6. Array Creation Functions
#=========================

arr6 = np.zeros(5)
arr7 = np.ones(5)
arr8 = np.arange(1,10)
arr9 = np.linspace(0,1,5)

print(arr6)
print(arr7)
print(arr8)
print(arr9)