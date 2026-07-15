"""
Topic: Pandas Series

Description:
This file contains my practice exercises for creating and working with Pandas Series.
"""

import pandas as pd

#=========================
#1. Creating a Series
#=========================
# Create a Pandas Series.
numbers = [10, 20, 30, 40, 50]
series1 = pd.Series(numbers)
print(series1)

#=========================
#2. Series Information
#=========================
# Print Series information.
print(series1.dtype)
print(series1.shape)
print(series1.size)

#=========================
#3. Accessing Elements
#=========================
# Access Series elements.
print(series1[0])
print(series1.iloc[-1])
print(series1[2])

#=========================
#4. Custom Index
#=========================

numbers = [85, 92, 78, 95]
grades = pd.Series(numbers, index=["Ali", "Ayşe", "Mehmet", "Zeynep"])
print(grades)
print(grades["Ayşe"])

#=========================
#5. Basic Operations
#=========================
# Sort a Pandas Series.

updated_grades = grades + 5
print(updated_grades)
print("Average grade:", grades.mean())
print("Highest grade:", grades.max())
print("Lowest grade:", grades.min())


#=========================
#6. Boolean Indexing
#=========================
# Filter Series values using a condition.

series = [50, 65, 80, 95, 100]
series2 = pd.Series(series)
filtered_grades = series2[series2 > 70]
print(filtered_grades)

#=========================
#7. Sorting a Series
#=========================
# Sort a Pandas Series.

sorted_grades =series2.sort_values()
print(sorted_grades)
reverse_sorted_grades = series2.sort_values(ascending = False)
print(reverse_sorted_grades)