import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Load data (numpy 28x28 bitmaps)
data_bird = np.load('bitmaps/bird.npy')
data_sheep = np.load('bitmaps/sheep.npy')
data_turtle = np.load('bitmaps/seaturtle.npy')

#Create label arrays
label_bird = ['bird'] * len(data_bird)
label_sheep = ['sheep'] * len(data_sheep)
label_turtle = ['turtle'] * len(data_turtle)


#Separating training data and testing data


# bird_train, bird_test = data_bird[:60000], data_bird[60000:70000]
# y_train, y_test = y[:60000], y[60000:70000]

# X_train, X_test = X[:60000], X[60000:70000]
# y_train, y_test = y[:60000], y[60000:70000]

# X_train, X_test = X[:60000], X[60000:70000]
# y_train, y_test = y[:60000], y[60000:70000]

print('Bird data length: ', len(data_bird))
print('Bird label data length: ', len(label_bird))
print('Sheep data length: ', len(data_sheep))
print('Sheep label data length: ', len(label_sheep))
print('Turtle data length: ', len(data_turtle))
print('Turtle label data length: ', len(label_turtle))

index_start = 10000 # <--- Change this for a different doodle
img1 = data_bird[index_start]
img2 = data_sheep[index_start]
img3 = data_turtle[index_start]

# plt.imshow(img1.reshape((28,28)))
# plt.show()
# plt.imshow(img2.reshape((28,28)))
# plt.show()
# plt.imshow(img3.reshape((28,28)))
# plt.show()


# concatinate

# print(len(data))
# datatot = pd.concat([data[:],data[:]])

# print(len(datatot))