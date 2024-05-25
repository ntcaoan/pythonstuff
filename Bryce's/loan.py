import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers


def acc_chart(results):
    plt.title("Accuracy Graph")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()


def loss_chart(results):
    plt.title("Loss Graph")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()


def do_graph_stuff(loan_data):
    """"""


loan_data = pd.read_csv('../Data/loan.csv')

# checking the data from csv file
print(loan_data.head(15).to_string())

# the occupation field is not needed for this
loan_data.drop(columns=['occupation'], inplace=True)
loan_copy = loan_data.copy()

# Categorized Strings should be converted to integer Values for the analysis
# loan_copy['gender'] = loan_copy['gender'].map({0: "Male", 1: "Female"})
# loan_copy['marital_status'] = loan_copy['marital_status'].map({0: "Married", 1: "Single"})
# loan_copy['loan_status'] = loan_copy['loan_status'].map({0: "Approved", 1: "Denied"})

# The Categorized Strings should be converted to integer Vales for the analysis.
loan_copy['gender'] = loan_copy['gender'].map({"Male": 0, "Female": 1})
loan_copy['marital_status'] = loan_copy['marital_status'].map({"Single": 0, "Married": 1})
loan_copy['loan_status'] = loan_copy['loan_status'].map({"Denied": 0, "Approved": 1})

# The education level should be broken up into 3 different types.
loan_copy['education_level'] = loan_copy['education_level'].map(
    {"High School": 0, "Bachelor's": 1, "Associate's": 1, "Master's": 2, "Doctoral": 2, })

# education level be broken to 3 types: High School, Bachelor's, Graduate Level
# def categorize_education(education):
#     if "High School" in education:
#         return 2
#     elif "Bachelor's" in education:
#         return 0
#     else:
#         return 1
#
#
# loan_copy['education_level'] = loan_copy['education_level'].apply(categorize_education)
#
# # Print the DataFrame to check the categorization
# print(loan_copy)
#
# # create separate DataFrames for each category:
# bachelor_data = loan_copy[loan_copy['education_level'] == 0]
# graduate_data = loan_copy[loan_copy['education_level'] == 1]
# high_school_data = loan_copy[loan_copy['education_level'] == 2]
#
# # Print the separate DataFrames to check
# print(f"Bachelor's Data: \n{bachelor_data}")
# print(f"Graduate Level Data: \n{graduate_data}")
# print(f"High School Data: \n{high_school_data}")

# (3) Graphing
# a) Heatmap for the given Dataframe
sns.heatmap(loan_copy.corr(), annot=True)
plt.show()


# b) Histograms:
# age against approved/denied
def age_histograms(loan_data):
    # separate the data based on loan status
    approved_data = loan_data[loan_data['loan_status'] == "Approved"]
    denied_data = loan_data[loan_data['loan_status'] == "Denied"]

    # plot histograms for approved loans
    plt.figure(figsize=(10, 5))
    plt.hist(approved_data['age'], bins=20, alpha=0.5, color='green', edgecolor='black', label='Approved')

    # plot histograms for denied loans
    plt.hist(denied_data['age'], bins=20, alpha=0.5, color='red', edgecolor='black', label='Denied')

    plt.legend()
    plt.title('Histogram of Age for Approved and Denied Loans')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()


# education against approved/denied
def education_histograms(loan_data):
    # separate the data based on loan status
    approved_data = loan_data[loan_data['loan_status'] == "Approved"]
    denied_data = loan_data[loan_data['loan_status'] == "Denied"]

    plt.figure(figsize=(10, 5))

    # plot histograms for approved loans
    plt.hist(approved_data['education_level'], bins=3, alpha=0.5, color='green', label='Approved', edgecolor='black')

    # plot histograms for denied loans
    plt.hist(denied_data['education_level'], bins=3, alpha=0.5, color='red', label='Denied', edgecolor='black')

    plt.legend()
    plt.title('Histogram of Education Level for Approved and Denied Loans')
    plt.xlabel('Education Level')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1, 2], labels=['Bachelor\'s', 'Graduate Level', 'High School'])
    plt.show()


# married/single status against approved/denied
def marital_status_histograms(loan_data):
    # separate the data based on loan status
    approved_data = loan_data[loan_data['loan_status'] == "Approved"]
    denied_data = loan_data[loan_data['loan_status'] == "Denied"]

    plt.figure(figsize=(10, 5))

    # approved loans
    plt.hist(approved_data['marital_status'], bins=3, alpha=0.5, color='green', label='Approved', edgecolor='black')
    plt.hist(denied_data['marital_status'], bins=3, alpha=0.5, color='red', label='Denied', edgecolor='black')

    plt.legend()
    plt.title("Histogram of Marital Status for Approved and Denied Loans")
    plt.xlabel("Marital Status")
    plt.ylabel("Frequency")
    plt.xticks(ticks=[0, 1], labels=['Married', 'Single'])
    plt.show()


age_histograms(loan_data)
education_histograms(loan_data)
marital_status_histograms(loan_data)

# (4) Create an appropriate mode given the Modified DataFrame you have prepared

# identify the X and Y (X is default for inputs, Y is default for output)
X = loan_copy.drop('loan_status', axis=1)
Y = loan_copy['loan_status']

print("Shape of X is %s" % str(X.shape))
print("Shape of Y is %s" % str(Y.shape))

model = models.Sequential()
model.add(layers.Dense(13, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# might need to use def acc_chart and loss_chart
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.22, epochs=100, batch_size=20)

acc_chart(history)
loss_chart(history)


# make some trained data about approved and denied loan status
