# %%
# Demonstrate loading in images

from pathlib import Path

import keras.src.backend
# PIL is the Python Imagine library - install pillow to get access to it
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as ts
from keras import layers
from keras import models, Sequential

# Create a path pointing to the directory we want to load
cat_dir = Path("./data/cat_dog_images/cat/")

# Want to see what directory we are actually pointing at
print(cat_dir.absolute())

# Let's check how many images there are in that directory
cat_files = list(cat_dir.glob("*.png"))
image_count = len(cat_files)
print(f"We have {image_count} pictures of cats")
print(cat_files)

# %%

# Let's load in one of those images using PIL
cat_image = Image.open(str(cat_files[0]))

print(f"The cat image size is {cat_image.size}, its type is {type(cat_image)}")
print(f"The cat image mode is {cat_image.mode}, and its format is {cat_image.format}")

plt.imshow(cat_image)
plt.show()

# %%
# We just want to treat our image as numeric data - so let's convert it to a numpy array!
cat_img_array = np.array(cat_image)
print("Cats as array:")
print(cat_img_array)
print(f"The shape of the cat array is {cat_img_array.shape}")

# %%
# Can also easily go the opposite direction - from a numpy array to an image
cat_img_array = cat_img_array[:, :, 0]
new_cat_img = Image.fromarray(cat_img_array)
plt.imshow(new_cat_img)
plt.show()

# %%
# This is a good time to examine things like the image format - like is it channel first
# or channel last? What does our ML library expect?
print(f"Keras expects images to be {keras.src.backend.image_data_format()}")
# If that wasn't what we wanted we could set it:
# keras.backend.set_image_data_format('channels_last')

# If our image was in the wrong shape, we could easily use numpy's roll function
# to change the order of the dimensions
rolled_cat_img_array = np.rollaxis(cat_img_array, 2, 0)
print(rolled_cat_img_array)
print(rolled_cat_img_array.shape)

# %%
# Since this is a numpy array, we can do whatever we can do to numpy arrays on it
# mutated_cat_array = np.where(cat_img_array < 100, 0, cat_img_array) # if cat_img_array less than 100 -> return 0 if not return cat_img_array
mutated_cat_array = np.where(cat_img_array < 100, 255, cat_img_array)
mutated_cat_array = mutated_cat_array * 0.25
mutated_cat_array = mutated_cat_array.astype(np.uint8)
mutated_cat_img = Image.fromarray(mutated_cat_array)
plt.imshow(mutated_cat_img)
plt.show()

# %%
# What if we wanted to save this resulting image?

# Let's first create a path for it:
mutated_cat_file = Path("./data/mutated_images/cat_1_mutated.png")
print(f"I might need to make my parent directory: {mutated_cat_file.parents[0]}")
mutated_cat_file.parents[0].mkdir(exist_ok=True)
print(
    f"The name of my mutated_cat_file is {mutated_cat_file.name} and its absolute path is {mutated_cat_file.absolute()}")

# To actually save it, just use Image.save
mutated_cat_img.save(mutated_cat_file, "PNG")

# %%

## DATA AUGMENTATION ##

# How can keras do image augmentation for us?
# Keras has a bunch of random layers, and we can just pass stuff through
# Tensorflow expects an id array of images, so it won't like just one single cat
cat_images_array = cat_img_array.reshape((1,) + cat_img_array.shape)
print(cat_images_array.shape)

# Let's add some of these keras preprocessing layers:
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal', input_shape=(cat_img_array.shape)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0, 1)
    ]
)

# Let's generate a few random variations
for i in range(9):
    augmented_images = image_augmentation(cat_images_array)
    plt.imshow(augmented_images[0].numpy().astype(np.uint8))
    plt.show()
    print(i)

# %%


# %%
