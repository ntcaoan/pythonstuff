import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import models, layers

loan_data = pd.read_csv('../Data/loan.csv')

# checking the data from csv file
print(loan_data.head().to_string())

# the occupation field is not needed for this
loan_data.drop(columns=['occupation'], inplace=True)
loan_copy = loan_data.copy()

# Categorized Strings should be converted to integer Values for the analysis
loan_copy['gender'] = loan_copy['gender'].map({0: "Male", 1: "Female"})
male_data = loan_copy[loan_copy['gender'] == "Male"]
female_data = loan_copy[loan_copy['gender'] == "Female"]


# education level be broken to 3 types: High School, Bachelor's, Graduate Level
def categorize_education(education):
    if "High School" in education:
        return 2
    elif "Bachelor's" in education:
        return 0
    else:
        return 1


loan_copy['education_level'] = loan_copy['education_level'].apply(categorize_education)

# Print the DataFrame to check the categorization
print(loan_copy)

# If you need to create separate DataFrames for each category:
bachelor_data = loan_copy[loan_copy['education_level'] == 0]
graduate_data = loan_copy[loan_copy['education_level'] == 1]
high_school_data = loan_copy[loan_copy['education_level'] == 2]

# Print the separate DataFrames to check
print(f"Bachelor's Data: \n{bachelor_data}")

print(f"Graduate Level Data: \n{graduate_data}")

print(f"High School Data: \n{high_school_data}")

# (3) Graphing
# a. Heatmap for the given Dataframe
numeric_columns = loan_copy.select_dtypes(include=[int, int, int, int, float, int, int]).columns
sns.heatmap(loan_copy[numeric_columns].corr(), annot=True)
plt.show()
