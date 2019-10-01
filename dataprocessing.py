import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


# Load data (numpy 28x28 bitmaps)
data_bird = np.load('bitmaps/bird.npy')
data_sheep = np.load('bitmaps/sheep.npy')
data_turtle = np.load('bitmaps/seaturtle.npy')

#Create label arrays
label_bird = ['bird'] * len(data_bird)
label_sheep = ['sheep'] * len(data_sheep)
label_turtle = ['turtle'] * len(data_turtle)


#Separating training data and testing data

#bird
bird_x_train, bird_x_test = data_bird[:20000], data_bird[20000:25000]
bird_y_train, bird_y_test = label_bird[:20000], label_bird[20000:25000]

#sheep
sheep_x_train, sheep_x_test = data_sheep[:20000], data_sheep[20000:25000]
sheep_y_train, sheep_y_test = label_sheep[:20000], label_sheep[20000:25000]

#turtle
turtle_x_train, turtle_x_test = data_turtle[:20000], data_turtle[20000:25000]
turtle_y_train, turtle_y_test = label_turtle[:20000], label_turtle[20000:25000]


# concatenate
x_train = np.concatenate([bird_x_train[:],sheep_x_train[:], turtle_x_train[:]])
y_train = np.concatenate([bird_y_train[:],sheep_y_train[:], turtle_y_train[:]])

x_test = np.concatenate([bird_x_test[:],sheep_x_test[:], turtle_x_test[:]])
y_test = np.concatenate([bird_y_test[:],sheep_y_test[:], turtle_y_test[:]])

#Shuffle the order
x_train, y_train = shuffle(x_train, y_train)

########################################################################################
####################################  Plotting #########################################
########################################################################################

index_start = 1000 # <--- Change this for a different doodle
img1 = x_train[index_start]
# img2 = data_sheep[index_start]
# img3 = data_turtle[index_start]

plt.imshow(img1.reshape((28,28)))
plt.title(y_train[index_start])
plt.show()
# plt.imshow(img2.reshape((28,28)))
# plt.show()
# plt.imshow(img3.reshape((28,28)))
# plt.show()

