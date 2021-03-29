import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model, Sequential
from keras.layers import Flatten, Dense

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
train_img = train_img / 255.0
test_img = test_img / 255.0

seq_model = load_model("my_model")  # load model

# Evaluate Loss and Accuracy of model
eval = seq_model.evaluate(x=test_img, y=test_lab, verbose=2)
print("Test Loss:", eval[0])
print("Test Accuracy:", eval[1])

predictions = seq_model.predict(test_img)  # predictions from the test data



from keras.preprocessing.image import load_img, img_to_array


# load and prepare the image
def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28)
    img = img.astype('float32')
    img = img / 255
    return img


