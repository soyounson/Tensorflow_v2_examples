#|****************************************************************|#
#|********** 02_Linear regression (tensorflow v.2.x) *************|#
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
#|     original written date :  Nov/21/2020                       |# 
#|                                                                |# 
#|****************************************************************|#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#::::::::::::::::::::::
# 1) SQUENTIAL MODEL
#::::::::::::::::::::::
#**********************
#\\\\\ Generate data
#**********************
x_data = np.array([1,2,3,4,5,6])
y_data = np.array([3,4,5,6,7,8])
#**********************
#\\\\\ Build a model
#**********************
model = Sequential()
# add input layer
model.add(Flatten(input_shape=(1,)))
# add output layer
model.add(Dense(1,activation='linear'))
#**********************
#\\\\\Compile a model 
#**********************
model.compile(optimizer=SGD(learning_rate=1e-2),loss='mse')
# in keras, we can check model before training
model.summary()
#**********************
#\\\\\ Train a model
#**********************
hist = model.fit(x_data,y_data, epochs=1000)
#**********************
#\\\\\ Predict
#**********************
# generate random value in a range from -5 to 10
x_pre = np.random.normal(-5,10,x_data.shape[0])
y_pre = model.predict(x_pre)
print(y_pre)

#::::::::::::::::::::::
# 2) LINEARL REGRESSION
#::::::::::::::::::::::
#**********************
#\\\\\ Generate data
#**********************
x_train = np.array([[1,2,0],[5,4,3],[1,2,-1],[3,1,0],[2,4,2],
                  [4,1,2],[-1,3,2],[4,3,3],[0,2,6],[2,2,1],
                  [1,-2,-2],[0,1,3],[1,1,3],[0,1,4],[2,3,3]])
y_train = np.array([-4,4,-6,3,-4,
                   9,-7,5,6,0,
                   4,3,5,5,1])

#**********************
#\\\\\ Build a model
#**********************
model = Sequential()
# consider linear regression model➔ activation function = linear
model.add(Dense(1,input_shape=(x_train.shape[1],),activation='linear'))
#**********************
#\\\\\Compile a model 
#**********************
# learning algorithm : Stocastic Gradient Descent algorithm
# loss function : Mean Square Error (MSE)
model.compile(optimizer=SGD(learning_rate=1e-2),loss='mse')
model.summary()
# Q) Why total parameters are equal to 4? 
#   : 3 weights (input #) + 1 bias (output #)
#**********************
#\\\\\ Train a model
#**********************
hist = model.fit(x_train,y_train,epochs=1000)
#**********************
#\\\\\ Evaluate a model
#**********************
# evaluate and predict a model
x_test = np.array([[5,5,0],[2,3,1],[-1,0,-1],[10,5,2],[4,-1,-2]])
# if we have exact answer
y_exact = np.array([2*data[0]-3*data[1]+2*data[2] for data in x_test])
y_pred = model.predict(x_test)
print('---------------------------------')
print('# predicted values: ')
print('---------------------------------')
print(y_pred)
print('---------------------------------')
print('# exact values:')
print('---------------------------------')
print(y_exact)
print('---------------------------------')
print('# error, y_exact - y_pred')
print('---------------------------------')
for i in range(y_exact.shape[0]):
    error = y_exact[i]-y_pred[i]
    print(error)
#**********************
#\\\\\ Check a model
#**********************
print(model.input)
print(model.output)
print(model.weights)
#**********************
#\\\\\ Plot
#**********************
fig00 = plt.figure(figsize=(6,6))
plt.rc('font',family='Times New Roman')
plt.plot(hist.history['loss'],label='train_loss',c='k')
plt.legend()
plt.title('Temporal evolution of loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(False)
# save 
fig00.savefig("02_temporal_evolution_of_loss_LR.pdf", bbox_inches='tight')    