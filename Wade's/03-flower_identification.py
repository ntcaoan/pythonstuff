from pathlib import Path
import keras
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

# Set some constants for our images:
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Let's set the path for where the model will be:
model_path = Path("./data/models/flower_model.keras")
flower_model = keras.models.load_model(model_path)
print(flower_model.summary())

# We will loop asking the user the path

while True:
    flower_file = input("Enter the path of an image you'd like us to identify:\n")
    # ./img/roses.jpg

    flower_file_path = Path(flower_file)
    if not flower_file_path.is_file():
        print(f"{flower_file} is not a file. Please enter the file you'd like us to identify")
        continue
    try:
        img = keras.utils.load_img(flower_file_path,
                                   target_size=(IMG_HEIGHT, IMG_WIDTH))
    except UnidentifiedImageError as e:
        print("Could not load in image - r u sure it is a valid img?")
        continue

    # Let's create a 4d array from our 3d image data to make it into a batch of one
    img_array = np.array(img).reshape((1,) + (IMG_HEIGHT, IMG_WIDTH, 3))

    # Let's get the predictions from the model:
    predictions = flower_model.predict(img_array)
    print(predictions)

    # Let's get the score for the first image:
    score = tf.nn.softmax(predictions[0])
    print(score)
    probable_class = CLASSES[np.argmax(score)]
    confidence = round(100 * np.max(score), 2)
    print(f"This image most likely belongs to class {probable_class} with a {confidence}% confidence")

