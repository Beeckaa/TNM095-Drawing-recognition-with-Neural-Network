import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from keras import utils as np_utils

# Load data (numpy 28x28 bitmaps)
data_bird = np.load('bitmaps/bird.npy')
data_sheep = np.load('bitmaps/sheep.npy')
data_turtle = np.load('bitmaps/seaturtle.npy')
data_hedgehog = np.load('bitmaps/hedgehog.npy')
data_octopus = np.load('bitmaps/octopus.npy')
data_giraffe = np.load('bitmaps/giraffe.npy')
data_cat = np.load('bitmaps/cat.npy')
data_fish = np.load('bitmaps/fish.npy')
data_butterfly = np.load('bitmaps/butterfly.npy')
data_lion = np.load('bitmaps/lion.npy')

# Create label arrays
label_bird = [0] * len(data_bird)
label_sheep = [1] * len(data_sheep)
label_turtle = [2] * len(data_turtle)
label_hedgehog = [3] * len(data_hedgehog)
label_octopus = [4] * len(data_octopus)
label_giraffe = [5] * len(data_giraffe)
label_cat = [6] * len(data_cat)
label_fish = [7] * len(data_fish)
label_butterfly = [8] * len(data_butterfly)
label_lion = [9] * len(data_lion)

# Separating training data and testing data

# Bird
bird_x_train, bird_x_test = data_bird[:50000], data_bird[50000:55000]
bird_y_train, bird_y_test = label_bird[:50000], label_bird[50000:55000]

# Sheep
sheep_x_train, sheep_x_test = data_sheep[:50000], data_sheep[50000:55000]
sheep_y_train, sheep_y_test = label_sheep[:50000], label_sheep[50000:55000]

# Turtle
turtle_x_train, turtle_x_test = data_turtle[:50000], data_turtle[50000:55000]
turtle_y_train, turtle_y_test = label_turtle[:50000], label_turtle[50000:55000]

# Hedgehog
hedgehog_x_train, hedgehog_x_test = data_hedgehog[:50000], data_hedgehog[50000:55000]
hedgehog_y_train, hedgehog_y_test = label_hedgehog[:50000], label_hedgehog[50000:55000]

# Octopus
octopus_x_train, octopus_x_test = data_octopus[:50000], data_octopus[50000:55000]
octopus_y_train, octopus_y_test = label_octopus[:50000], label_octopus[50000:55000]

# Giraffe
giraffe_x_train, giraffe_x_test = data_giraffe[:50000], data_giraffe[50000:55000]
giraffe_y_train, giraffe_y_test = label_giraffe[:50000], label_giraffe[50000:55000]

# Cat
cat_x_train, cat_x_test = data_cat[:50000], data_cat[50000:55000]
cat_y_train, cat_y_test = label_cat[:50000], label_cat[50000:55000]

# Fish
fish_x_train, fish_x_test = data_fish[:50000], data_fish[50000:55000]
fish_y_train, fish_y_test = label_fish[:50000], label_fish[50000:55000]

# Butterfly
butterfly_x_train, butterfly_x_test = data_butterfly[:50000], data_butterfly[50000:55000]
butterfly_y_train, butterfly_y_test = label_butterfly[:50000], label_butterfly[50000:55000]

# Lion
lion_x_train, lion_x_test = data_lion[:50000], data_lion[50000:55000]
lion_y_train, lion_y_test = label_lion[:50000], label_lion[50000:55000]

# concatenate
x_train = np.concatenate([bird_x_train[:], sheep_x_train[:], turtle_x_train[:], hedgehog_x_train[:], octopus_x_train[:], giraffe_x_train[:], cat_x_train[:], fish_x_train[:], butterfly_x_train[:], lion_x_train[:]])
y_train = np.concatenate([bird_y_train[:], sheep_y_train[:], turtle_y_train[:], hedgehog_y_train[:], octopus_y_train[:], giraffe_y_train[:], cat_y_train[:], fish_y_train[:], butterfly_y_train[:], lion_y_train[:]])

x_test = np.concatenate([bird_x_test[:], sheep_x_test[:], turtle_x_test[:], hedgehog_x_test[:], octopus_x_test[:], giraffe_x_test[:], cat_x_test[:], fish_x_test[:], butterfly_x_test[:], lion_x_test[:]])
y_test = np.concatenate([bird_y_test[:], sheep_y_test[:], turtle_y_test[:], hedgehog_y_test[:], octopus_y_test[:], giraffe_y_test[:], cat_y_test[:], fish_y_test[:], butterfly_y_test[:], lion_y_test[:]])

# Shuffle the order
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

# Normalizing the images
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0

# # Creating a simple model
# model = tf.keras.models.Sequential()
# #model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# #model.add(tf.keras.layers.Dense(128, input_shape=(784,), activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, input_shape=(28, 28), activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Convolution2D(16, (5, 5),
                        padding='same',
                        input_shape=x_train.shape[1:], activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Convolution2D(32, (5, 5), padding='same', activation= 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Convolution2D(64, (5, 5), padding='same', activation= 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size =(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) 

#Tensorboard init
tensorboard = TensorBoard(log_dir='logs\{}'.format(time()))

# Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.fit(np.array(x_train), np.array(y_train), epochs=2, callbacks=[tensorboard])
model.fit(x = x_train, y = y_train, validation_split=0.1, verbose=2, epochs=8, callbacks=[tensorboard])

# Testing the model
val_loss, val_acc = model.evaluate(np.array(x_test), np.array(y_test))
print(val_loss, val_acc)

# Save the model an loading it into the program again. Also saving it in js format
#model.save('doodle_model')
#new_model = tf.keras.models.load_model('doodle_model')
tfjs.converters.save_keras_model(model, "doodle_model_js")

# Predict (Predicts all images in x_test and creates an array with each prediction)
predictions = model.predict(x_test)

########################################################################################
#################################### Plotting ##########################################
########################################################################################

#Plot the first 25 entries accompanied with a label-prediction for each one
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img1 = x_test[i]
    plt.imshow(img1.reshape((28,28)), cmap=plt.cm.binary)
    if np.argmax(predictions[[i]]) == 0:
        plt.xlabel('bird ')
    elif np.argmax(predictions[[i]]) == 1:
        plt.xlabel('sheep ')
    elif np.argmax(predictions[[i]]) == 2:
        plt.xlabel('turtle ')
    elif np.argmax(predictions[[i]]) == 3:
        plt.xlabel('hedgehog ')
    elif np.argmax(predictions[[i]]) == 4:
        plt.xlabel('octopus ')
    elif np.argmax(predictions[[i]]) == 5:
        plt.xlabel('giraffe ')
    elif np.argmax(predictions[[i]]) == 6:
        plt.xlabel('octopus ')
    elif np.argmax(predictions[[i]]) == 7:
        plt.xlabel('giraffe ')
    elif np.argmax(predictions[[i]]) == 8:
        plt.xlabel('butterfly ')
    elif np.argmax(predictions[[i]]) == 9:
        plt.xlabel('lion ')
    #plt.ylabel(y_test[i])
    plt.ylabel(round(100*np.max(predictions[i]), 2))
plt.show()


########################################################################################
################################## Confusion Matrix ###################################
########################################################################################

#Create a new array and insert the predicted values
temp_list = []

for i in range(len(y_test)):
    if np.argmax(predictions[[i]]) == 0:
        temp_list.append(0)
        
    elif np.argmax(predictions[[i]]) == 1:
        temp_list.append(1)
       
    elif np.argmax(predictions[[i]]) == 2:
        temp_list.append(2)

    elif np.argmax(predictions[[i]]) == 3:
        temp_list.append(3)

    elif np.argmax(predictions[[i]]) == 4:
        temp_list.append(4)

    elif np.argmax(predictions[[i]]) == 5:
        temp_list.append(5)

    elif np.argmax(predictions[[i]]) == 6:
        temp_list.append(6)

    elif np.argmax(predictions[[i]]) == 7:
        temp_list.append(7)

    elif np.argmax(predictions[[i]]) == 8:
        temp_list.append(8)

    elif np.argmax(predictions[[i]]) == 9:
        temp_list.append(9)
        

#Create a confusion matrix
cm = confusion_matrix(y_test, temp_list)

animals = ['bird', 'sheep', 'turtle', 'hedgehog','octopus', 'giraffe', 'cat', 'fish', 'butterfly', 'lion']

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(len(animals)),
           yticks=np.arange((len(animals))),
           xticklabels=animals, 
           yticklabels=animals,
           ylim=[-0.5,10],
           title='Confusion Matrix',
           xlabel= 'Predicted animal',
           ylabel='True animal')

#Rotate the x-tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.show()
