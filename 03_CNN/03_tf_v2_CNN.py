#|****************************************************************|#
#|************* 03_image + CNN (tensorflow v.2.x) ****************|#
#|****************************************************************|# 
#|                                                                |#
#|     prerequisite : python v.3.x                                |#
#|                    tensorflow v.2.x                            |# 
#|                                                                |#
#|     shortcut to run a code : ctl + enter                       |#
#|                                                                |#
#|----------------------------------------------------------------|#
#|     written by: S.Son (soyoun.son@gmail.com)                   |# 
#|                 https://github.com/soyounson                   |# 
#|                                                                |# 
#|     original written date :  Nov/30/2020                       |# 
#|                                                                |# 
#|****************************************************************|#

# ref : https://www.tensorflow.org/tutorials/images/cnn

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#**********************
#\\\\\ Load data
#**********************
mnist = tf.keras.datasets.mnist

#______________________
# total 70,000 images 
# resolution 28×28 pixels
# Features 
# train_images : (60000, 28, 28) 
# test_images : (10000, 28, 28)
# Label (an array of intergers, ranging from 0 to 9)
# train_labels : 60000
# test_labels : 10000
#______________________
# divide data into train + test data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images0, train_labels0 = train_images.copy(), train_labels.copy()
# images have 3 channels (RGB) but, MNIST dataset has only one channel (gray).
# add a channel dimension
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# or 
# train_images = train_images[..., tf.newaxis]
# test_images = test_images[..., tf.newaxis]

#**********************
#\\\\\ Preprocess
#
# Scale these values to a range of 0 to 1 
# before feeding them to the neural network model. 
# To do so, divide the values by 255 (= np.max(train_images)). 
#**********************
train_images, test_images = train_images/ 255.0, test_images / 255.0
# or 
# train_images, test_images = train_images/np.max(train_images), test_images/np.max(test_images)

#**********************
#\\\\\ Label (0~9)
#**********************
class_names = ['zero','one','two','three','four','five','six',
               'seven','eight','nine']

#**********************
#\\\\\ Build a model 
#**********************
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# dense layer accepts only 1D 
# so Flatten layer is required (3D → 1D)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# 10 outputs 
model.add(layers.Dense(10, activation='softmax'))
#**********************
#\\\\\ Architecture of our model 
#**********************
model.summary()

#**********************
#\\\\\ Compile a model 
#**********************
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

#**********************
#\\\\\ Evaluate 
#**********************
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#**********************
#\\\\\ Check weights
#**********************
print('======================================')
print('input dimension =', model.input.shape)
print('======================================')
print('output dimension =', model.output.shape)
print('======================================')
print(model.weights)