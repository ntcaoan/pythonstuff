import numpy

import pandas as pd
import numpy as np
from numpy import array
import seaborn as sb
import matplotlib as plt
import keras


def acc_chart(results):
    plt.pyplot.title("Accuracy Graph")
    plt.pyplot.ylabel("accuracy")
    plt.pyplot.xlabel("epoch")
    plt.pyplot.plot(results.history['accuracy'])
    plt.pyplot.plot(results.history['val_accuracy'])
    plt.pyplot.legend(['train', 'test'], loc="upper left")
    plt.pyplot.show()


def loss_chart(results):
    plt.pyplot.title("Loss Graph")
    plt.pyplot.ylabel("Loss")
    plt.pyplot.xlabel("epoch")
    plt.pyplot.plot(results.history['loss'])
    plt.pyplot.plot(results.history['val_loss'])
    plt.pyplot.legend(['train', 'test'], loc="upper left")
    plt.pyplot.show()


df = pd.read_csv("Data/kc_house_data.csv")

print(df.head().to_string())

print("\nShape and size")
print(df.shape)

print(df.loc[:0, "date"])

# Get the year out of this...
# Create a new series that will be called registration year and for our
# purposes we want this to be an integer
df['reg_year'] = df['date'].str[:4]

# we want to convert this to be an int
df['reg_year'] = df['reg_year'].astype('int')

print(df.head().to_string())
print(df.dtypes)

# We want to create a new series (house_age) that will
# be the difference between the reg_year and either the yr_built
# or the yr_rennovated

# create a new series - initially set all values to be Nan
df['house_age'] = np.NAN

for i, j in enumerate(df['yr_renovated']):
    if j == 0:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_built']
    else:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_renovated']

print(df.head().to_string())

# Once this calculation is done - we don't care about the other dates - let's drop them from the table
df.drop(["date", "yr_built", "yr_renovated", 'reg_year'], axis=1, inplace=True)
df.drop(["zipcode", "lat", "long", "id"], axis=1, inplace=True)
df.drop(['house_age', 'sqft_lot15', 'condition', 'sqft_lot'], axis=1, inplace=True)

print(df.dtypes)
print(df.head().to_string())

# Normally you would have to set up a series of tests up to see if there are any odd values in our
# existing DataSet - The Authors for this example pointed out that some of the house ages are -1
# That is a bad data value

# df_bad = df[df["house_age"] < 0]
# print("\n\nBad Data Points")
# print(df_bad.to_string())

# df = df[df["house_age"] >= 0]
# df_bad = df[df["house_age"] < 0]

# print("\n\nAny Entries left??")
# print(df_bad.to_string())

# Let's look at all the series to start with
# for i in df.columns:
#     sb.displot(df[i])
#     plt.pyplot.show()

# Mapp all of the series against each other
# sb.pairplot(df)
# plt.pyplot.show()

plt.pyplot.figure(figsize=(20, 10))
sb.heatmap(df.corr(), annot=True)
plt.pyplot.show()


# We want to train our model to calculate the
# expected price built upon the other parameters
# For this purpose we want to specify our inputs as anything that is not the price
# We want output to be the price

# By convention X is supposed to be this inputs
# Y is supposed to be the array of outputs
X = df.drop('price', axis=1)
Y = df['price']

print(X.head().to_string())
print(Y.head().to_string())

# 5 - 17 - 2024
my_model = keras.Sequential()

# add input layer
my_model.add(keras.layers.Dense(14, activation='relu'))
my_model.add(keras.layers.Dense(4, activation='relu'))
my_model.add(keras.layers.Dense(2, activation='relu'))
my_model.add(keras.layers.Dense(1))

my_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
results = my_model.fit(X, Y, validation_split=0.33, batch_size=30, epochs=80)

acc_chart(results)
loss_chart(results)

# We are going to do some predictive results
# samp_house = array([
#     [2, 3, 1280, 5550, 1, 0, 0, 4, 7, 1280, 800, 1440, 5750, 35]
# ])
# samp_house = numpy.array(samp_house, dtype=numpy.float64)
#
# est_price = my_model.predict(samp_house)
# print(est_price[0])

print(df.loc[75:75].to_string())
