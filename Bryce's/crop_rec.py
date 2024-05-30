import pandas as pd
import keras
from keras import layers, Sequential, models
import numpy as np
import matplotlib.pyplot as plt


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


# identify the top 2 crops for a specified Crop
def identify_top2_crops(predictions, crop_labels):
    top_indices = np.argsort(predictions)[-2:][::-1]
    return [crop_labels[indice] for indice in top_indices]


crop_rec_data = pd.read_csv('../Data/Crop_Recommendation.csv')

print(crop_rec_data.head(13).to_string())

# since the Crop is string -> change to number
# map_dict = {'Rice': 0, 'Maize': 1, 'ChickPea': 2, 'KidneyBeans': 3, 'PigeonPeas': 4,
#             'MothBeans': 5, 'MungBean': 6, 'Blackgram': 7, 'Lentil': 8, 'Pomegranate': 9,
#             'Banana': 10, 'Mango': 11, 'Grapes': 12, 'Watermelon': 13, 'Muskmelon': 14,
#             'Apple': 15, 'Orange': 16, 'Papaya': 17, 'Coconut': 18, 'Cotton': 19,
#             'Jute': 20, 'Coffee': 21}
#
# crop_rec_data['Crop'] = crop_rec_data['Crop'].map(map_dict)

map_dict = {crop: i for i, crop in enumerate(crop_rec_data['Crop'].unique())}
crop_rec_data['Crop'] = crop_rec_data['Crop'].map(map_dict)

X = crop_rec_data.drop('Crop', axis=1)
y = crop_rec_data['Crop']

print("Shape of X is %s" % str(X.shape))
print("Shape of y is %s" % str(y.shape))

model = models.Sequential()
model.add(layers.Dense(13, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dense(len(map_dict), activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X, y, epochs=50, validation_split=.2, batch_size=100)

acc_chart(history)
loss_chart(history)

# a sample trace of 3 predictions that u used to test ur model
crop_label = list(map_dict.keys())

# 1. model testing for Rice
md_rice = np.array([
    [90, 42, 43, 20.8, 82, 6.5, 202.9]
], dtype=np.float64)
rice_pred = model.predict(md_rice)
print("\nPredicted Rice sample:")
print(rice_pred)
# top 2 crops
top2_crops_rice = identify_top2_crops(rice_pred[0], crop_label)
print("\nTop 2 crops recommendation for Rice sample: ")
print(top2_crops_rice)

# 2. model testing for Maize
md_maize = np.array([
    [61, 44, 17, 26, 71.5, 6.9, 102]
], dtype=np.float64)
maize_pred = model.predict(md_maize)
print("\nPredicted Maize sample:")
print(maize_pred)
top2_crops_maize = identify_top2_crops(maize_pred[0], crop_label)
print("\nTop 2 crops recommendation for Maize sample:")
print(top2_crops_maize)

# 3. model testing for Mango
md_mango = np.array([
    [0, 21, 32, 35.8, 54, 6.4, 92]
], dtype=np.float64)
mango_pred = model.predict(md_mango)
print("\nPredicted Mango sample:")
print(mango_pred)
top2_crops_mango = identify_top2_crops(mango_pred[0], crop_label)
print("\nTop 2 crops recommendation for Mango sample:")
print(top2_crops_mango)
