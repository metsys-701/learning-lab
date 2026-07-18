"""
Topic: Pandas Reading and Writing Files

Description:
This file contains my practice exercises for reading, exploring, and writing data using Pandas.
"""

import pandas as pd

#=========================
#1. Creating a DataFrame
#=========================
#Create a sample DataFrame.

students_df = pd.DataFrame({
    'name' : ['Ali', 'Ayşe', 'Mehmet'],
    'age' : [20, 22, 21],
    'city' : ['Istanbul', 'Ankara', 'Izmir']
})

#=========================
#2. Saving to CSV
#=========================
#Save a DataFrame to a CSV file.

students_df.to_csv('student.csv', index = False)

#=========================
#3. Reading a CSV File
#=========================
#Read a CSV file.

students_df = pd.read_csv('student.csv')
print(students_df)

#=========================
#4. Exploring Data
#=========================
#Display basic information about the DataFrame.

print(students_df.head())
print(students_df.tail())
students_df.info()
print(students_df.describe())

#=========================
#5. Selecting Data
#=========================
#Select data from the DataFrame.

print("Names:", students_df['name'])
print("Cities:", students_df['city'])
print("First two rows:", students_df.head(2))
print("Last row:", students_df.tail(1))

#=========================
#6. Saving a Filtered DataFrame
#=========================
#Save filtered data to a new CSV file.
filtered_students = students_df[students_df['age'] >= 21]
print(filtered_students)
filtered_students.to_csv("filtered_students.csv", index=False)
