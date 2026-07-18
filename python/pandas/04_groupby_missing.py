
"""
Topic: Pandas GroupBy and Missing Values

Description:
This file contains my practice exercises for grouping data and handling missing values in Pandas.
"""

import pandas as pd
import numpy as np
#=========================
#1. Creating a DataFrame
#=========================

df =pd.DataFrame({
    'name': ['Ali', 'Ayşe', 'Mehmet', 'Zeynep', 'Ahmet'],
    'department':['IT', 'HR', 'IT', 'HR', 'IT'],
    'salary':[5000, 4500, 6000, 4800, 5500]

})

#=========================
#2. GroupBy Mean
#=========================
#Calculate the average salary for each department.

average_salary = df.groupby('department')['salary'].mean().reset_index()
print(average_salary)

#=========================
#3. GroupBy Count
#=========================
#Count employees in each department.

employee_count = df.groupby('department')['department'].count()
print(employee_count)

#=========================
#4. Missing Values
#=========================
#Create a DataFrame containing missing values.
scores_df = pd.DataFrame({
    'name': ['Ali', 'Ayşe', 'Mehmet','Zeynep'],
    'score' : [85, np.nan, 78, np.nan]
})

print(scores_df)

#=========================
#5. Detect Missing Values
#=========================
#Detect missing values.

print(scores_df.isnull())
print(scores_df.isnull().sum())

#=========================
#6. Fill Missing Values
#=========================
#Replace missing values.
filled_scores = scores_df.fillna(0)
print(filled_scores)

#=========================
#7. Remove Missing Values
#=========================
#Remove rows with missing values.

clean_scores = scores_df.dropna()
print(clean_scores)
