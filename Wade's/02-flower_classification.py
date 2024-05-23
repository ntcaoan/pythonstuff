# %%
# Load in our imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers, Sequential


# %%
# Let's look at the data we are working with

data_dir = Path("./data/flower_images")
print(data_dir.absolute())

image_count = len(list(data_dir.glob("*/*.jpg")))
print(f"There are {image_count} images")

# Let's grab just the roses:
roses = list(data_dir.glob('roses/*'))
for i in range(5):
    my_image = Image.open(str(roses[i]))
    plt.imshow(my_image)
    plt.show()

# %%
# There are a lot of images in there, so we'll want to use some of the keras utilities
# to load it in. This will allow us to load entire directories of images in just a few lines
# You can also do it from scratch using tensorflow's data module

# Let's set some parameters for the loader
BATCH_SIZE = 32
# Our sample images are not uniform, so let's impose some constraints on height and width
IMG_HEIGHT = 180
IMG_WIDTH = 180

# We'll also want to have a validation split - so in this case, we'll try using 80%
# of the images for training, and 20% for validation
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# It'll automatically use the directories as "class names" which is usually what we want
class_names = train_ds.class_names
print(class_names)

# %%
# Let's visualize the first 9 images from the training datasets:
plt.figure(figsize=(10, 10))

# Let's take 1 "batch" from our dataset:
for images, labels in train_ds.take(1):
    print("One image batch: ")
    print(images.shape)
    print("One label batch: ")
    print(labels.shape)

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

# %%
# Loading these images in, we do want to configure for performance
# There's lots of tweaking you can do, but 2 obvious ones are setting the cache size
# and doing perfect so the data preprocessing and the model execution overlap
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# We will also want to standardize the data
# The numbers in our RGB channels are going to be 1 byte - 0-255.
# That isn't ideal for neural networks - typically we want small input values between 0 and 1
# So when we build our model, we'll use a resclaing layer to standardize the values to be
# in the 0 to 1 range
normalization_layer = layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# We could apply this
# throw it in at the

# %%

num_classes = len(class_names)
# Building the model
model = Sequential([
    normalization_layer,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

# %%
# Now we can train the model:

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# %%
# Let's visualize results of our training so we can make some decisions based on it:

# Grab the accuracy values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Grab the loss values:
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="upper right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, labels='Validation Loss')
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

plt.show()
