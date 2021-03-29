import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
train_img = train_img/255.0
test_img = test_img/255.0

for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(train_img[i], cmap='gray_r')
    plt.title("Digit: {}".format(train_lab[i]))
    plt.subplots_adjust(hspace=0.5)
    plt.axis('off')
# plt.show()

plt.hist(train_img[0].reshape(784), facecolor='orange')
plt.title("Pixel vs. its intensity", fontsize=16)
plt.ylabel("Pixel")
plt.xlabel("Intensity")
# plt.show()

print("Training images shape:", train_img.shape)
print("Test images shape:", test_img.shape)

seq_model = Sequential()
input_layer = Flatten(input_shape=(28, 28))
seq_model.add(input_layer)
hidden_layer1 = Dense(512, activation='relu')
seq_model.add(hidden_layer1)
hidden_layer2 = Dense(512, activation='relu')
seq_model.add(hidden_layer2)
output_layer = Dense(10, activation='softmax')
seq_model.add(output_layer)


seq_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

print("Training")
seq_model.fit(train_img, train_lab, epochs=25, verbose=2)
seq_model.save("my_model", overwrite=True)
