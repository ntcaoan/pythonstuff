import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import models, layers

loan_data = pd.read_csv('../Data/loan.csv')

# checking the data from csv file
print(loan_data.head().to_string())

# the occupation field is not needed for this
loan_data.drop(columns=['occupation'], inplace=True)


# Categorized Strings should be converted to integer Values for the analysis


# education level be broken to 3 types: High School, Bachelor's, Graduate Level

loan_copy = loan_data.copy()
loan_copy['education_level'] = loan_copy['education_level'].map({0: "Bachelor's", 1: "Graduate Level", 2: "High School"})

bachelor_data = loan_copy[loan_copy['education_level'] == "Bachelor's"]
high_school = loan_copy[loan_copy['education_level'] == "High School"]
bachelor_data = loan_copy[loan_copy['education_level'] == "Master" and loan_copy['education_level'] == "Doctoral"]


print(bachelor_data)





# (3) Graphing
# a. Heatmap for the given Dataframe
# sns.heatmap(loan_data.corr(), annot=True)
# plt.show()


