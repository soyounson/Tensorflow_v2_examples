#|****************************************************************|#
#|*********** 01_math operations (tensorflow v.2.x) **************|#
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
#|     original written date :  Nov/20/2020                       |# 
#|                                                                |# 
#|****************************************************************|#
import tensorflow as tf
import tensorboard
#**********************
#\\\\\ check your tf version
#**********************
print('---------------------------------')
print('# Your tensorflow version:', tf.__version__)
print('---------------------------------')
#**********************
#\\\\\ Define constants
#********************** 
a = tf.constant(3.0, name='a')
b = tf.constant(5.0, name='b')
c = tf.constant(1.0, name='c')
#**********************
#\\\\\ Define variable
#**********************
f = tf.Variable(tf.random.normal([1]))
# print (tensor, numpy)
print('---------------------------------')
print('# check our variables')
print(f,f.numpy())
print('---------------------------------')
#**********************
#\\\\\ Operation
#**********************
d = tf.add(a,b).numpy()
# change type : numpy ➔ tensor 
e = tf.convert_to_tensor(d)
print('# check operation')
print(d, e)
print('---------------------------------')
#**********************
#\\\\\ Function
#**********************
@tf.function
def my_opt(x,y,i):
    return x+y*i

tot_iter = 3
for i in range(0,tot_iter):
    d = my_opt(a,b,i)
    print('# in a for loop:')
    print('step=',i,',value=',d.numpy())
    print('---------------------------------')