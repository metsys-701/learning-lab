"""
Topic: Pandas Selecting and Filtering

Description:
This file contains my practice exercises for selecting and filtering data in Pandas DataFrames.
"""

import pandas as pd

#=========================
#1. Selecting a Single Column
#=========================
# Select individual columns from a DataFrame.

student =pd.DataFrame({
    "name":['Ali', 'Ayşe', 'Mehmet', 'Zeynep'],
    "age":[20, 22, 21, 23],
    "score":[85, 92, 78, 95]
})

print("Name columns:\n", student["name"])
print("Score columns:\n", student["score"])

#=========================
#2. Selecting Multiple Columns
#=========================
# Select multiple columns from a DataFrame.

print("Name and Score columns:")
print(student[["name", "score"]])

#=========================
#3. Filtering Rows
#=========================
# Filter rows based on a condition.

print(student[student["score"]>80])

#=========================
#4. Multiple Conditions
#=========================
# Filter rows using multiple conditions.

print(student[(student["age"] >= 22) & (student["score"] >= 90)]["name"] )

#=========================
#5. Selecting Specific Columns After Filtering
#=========================
# Select specific columns after filtering rows.

print(student[student["age"] >= 22]["score"])

#=========================
#6. loc and iloc
#=========================
# Access data using loc and iloc.

print(student.loc[0,"name"])
print(student.iloc[1])
print(student.loc[0:,"score"])
print(student.iloc[:2])