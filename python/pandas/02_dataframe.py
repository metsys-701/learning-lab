"""
Topic: Pandas DataFrame

Description:
This file contains my practice exercises for creating and working with Pandas DataFrames.
"""
import pandas as pd

#=========================
#1. Creating a DataFrame
#=========================
# Create a Pandas DataFrame.

students_df = pd.DataFrame({
    'name':['Ali', 'Ayşe', 'Mehmet', 'Zeynep'],
    'age': [20, 22, 21, 23],
    'score': [85, 92, 78, 95]
})

print(students_df)

#=========================
#2. DataFrame Information
#=========================
# Print DataFrame information.

print("shape:",students_df.shape)
print("columns:", students_df.columns)
print("Data types:")
print(students_df.dtypes)
print("Number of rows:",students_df.shape[0])
print("Number of columns:",students_df.shape[1])

#=========================
#3. Selecting Columns
#=========================
# Select DataFrame columns.

print("Name column:")
print(students_df["name"])
print("Students column:", students_df["score"])
print("Name and score:", students_df[["name","score"]])

#=========================
#4. Accessing Rows
#=========================
# Access DataFrame rows.

print("first row:", students_df.iloc[0])
print("last row:", students_df.iloc[-1])
print("first two rows:", students_df.head(2))
print("last two rows:", students_df.tail(2))

#=========================
#5. Basic Statistics
#=========================
# Calculate basic statistics.

mean_grade = students_df["score"].mean()
print("mean grade:",mean_grade)
highest_grade = students_df["score"].max()
print("highest grade:", highest_grade)
lowest_grade = students_df["score"].min()
print("lowest grade:", lowest_grade)

#=========================
#6. Adding a New Column
#=========================
# Add a new column.

students_df["passed"] = (students_df["score"] >= 80).map({True: 'Yes', False: 'No'})
print(students_df)
