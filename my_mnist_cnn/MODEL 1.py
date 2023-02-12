# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 00:48:41 2023

@author: harry
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add random noise to the images
def add_noise(images):
    return images + np.random.normal(0, 0.1, size=images.shape)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# Plot an example of an image before and after noise
plt.subplot(1, 2, 1)
plt.imshow(x_train[3], cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(x_train_noisy[3], cmap='gray')
plt.title('Noisy Image')
plt.show()

# Reshape the data to (num_samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_train_noisy = x_train_noisy.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_test_noisy = x_test_noisy.reshape(-1, 28, 28, 1)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_noisy, y_train, epochs=10, validation_data=(x_test_noisy, y_test))

# Plot the accuracy and loss vs epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), acc, label='Training Accuracy')
plt.plot(range(1, 11), val_acc, label='Validation Accuracy')