import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras import models
from keras import layers



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


def do_graph_stuff(dfHeart):
    """"""
    dfMale = dfHeart[dfHeart['sex'] == "Male"]
    sns.countplot(x='condition', data=dfMale)
    plt.title("Ratios of Healthy Males to Heart Risk")
    plt.show()

    dfFemale = dfHeart[dfHeart['sex'] == "Female"]
    sns.countplot(x='condition', data=dfFemale)
    plt.title("Ratios of healthy Females to Heart Risk")
    plt.show()

    # Let's compare
    condHealth = dfHeart['condition'] == 'Healthy'
    condAtRisk = dfHeart['condition'] == 'Heart Risk'

    # Compare age with Health
    plt.hist(dfHeart[condHealth]['age'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['age'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Age and Heart Risk")
    plt.legend()
    plt.show()

    # Compare cholestoral with health
    plt.hist(dfHeart[condHealth]['chol'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['chol'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Chol and Heart Risk")
    plt.legend()
    plt.show()

    # Compare Max Heart Rate with Health
    plt.hist(dfHeart[condHealth]['thalach'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['thalach'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Max Heart Rate and Heart Risk")
    plt.legend()
    plt.show()


    # Identify all the entries in the outlier for Choelestoral
    # Create a dataSet just composed of High Cholestoral entries
    dfHG = dfHeart[dfHeart['chol'] > 500]
    print(dfHG.head().to_string())

    df_low_high = dfHeart[dfHeart['thalach'] < 80]
    print(df_low_high.to_string())


dfHeart = pd.read_csv("Data/heart_cleveland_upload.csv")

dfCopy = dfHeart.copy()

# Enhance the copy
dfCopy['condition'] = dfCopy['condition'].map({0: "Healthy", 1: "Heart Risk"})
dfCopy['sex'] = dfCopy['sex'].map({1: "Male", 0: "Female"})

print(dfCopy.head().to_string())

dfMale = dfCopy[dfCopy['sex'] == "Male"]
# do_graph_stuff(dfCopy)

dfHeart = dfHeart[dfHeart['chol'] < 500]
dfHeart = dfHeart[dfHeart['thalach'] > 80]

# Identify the X and Y (X is default for inputs, Y is default for output)
X = dfHeart.drop('condition', axis=1)
Y = dfHeart['condition']

print("Shape of X is %s" % str(X.shape))
print("Shape of Y is %s" % str(Y.shape))

model = models.Sequential()
model.add(layers.Dense(13, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.22, epochs=200)

acc_chart(history)
loss_chart(history)


X_at_Risk = np.array([
    [62, 1, 3, 145, 250, 1, 2, 120, 0, 1.4, 1, 1, 0]
], dtype=np.float64)
print(model.predict(X_at_Risk))

Y_at_Risk = (model.predict(X_at_Risk) > 0.5).astype(int)
print(Y_at_Risk[0])

X_Healthy = np.array([
    [50, 1, 2, 129, 196, 0, 0, 163, 0, 0, 0, 0, 0]
], dtype=np.float64)

Y_Healthy = (model.predict(X_Healthy) > 0.5).astype(int)
print(Y_Healthy[0])


