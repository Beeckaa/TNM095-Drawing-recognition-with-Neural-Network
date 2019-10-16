import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.python.util import deprecation
from keras.utils import plot_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import os


#Remove warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

########################################################################################
################################### Functions ##########################################
########################################################################################

def process_data(num_of_train, num_of_test):

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_train_reshape = []
    x_test_reshape = []
    x_train_norm = []
    x_test_norm = []

    # Load data (numpy 28x28 bitmaps)
    data_bird = np.load('bitmaps/bird.npy')
    data_sheep = np.load('bitmaps/sheep.npy')
    data_turtle = np.load('bitmaps/seaturtle.npy')
    data_hedgehog = np.load('bitmaps/hedgehog.npy')
    data_octopus = np.load('bitmaps/octopus.npy')
    data_giraffe = np.load('bitmaps/giraffe.npy')
    data_cat = np.load('bitmaps/cat.npy')
    data_butterfly = np.load('bitmaps/butterfly.npy')
    data_lion = np.load('bitmaps/lion.npy')
    data_fish = np.load('bitmaps/fish.npy')

    # Create label arrays
    label_bird = [0] * len(data_bird)
    label_sheep = [1] * len(data_sheep)
    label_turtle = [2] * len(data_turtle)
    label_hedgehog = [3] * len(data_hedgehog)
    label_octopus = [4] * len(data_octopus)
    label_giraffe = [5] * len(data_giraffe)
    label_cat = [6] * len(data_cat)
    label_butterfly = [7] * len(data_butterfly)
    label_lion = [8] * len(data_lion)
    label_fish = [9] * len(data_fish)

    # Bird
    bird_x_train, bird_x_test = data_bird[:num_of_train], data_bird[num_of_train:(num_of_train + num_of_test)]
    bird_y_train, bird_y_test = label_bird[:num_of_train], label_bird[num_of_train:(num_of_train + num_of_test)]

    # Sheep
    sheep_x_train, sheep_x_test = data_sheep[:num_of_train], data_sheep[num_of_train:(num_of_train + num_of_test)]
    sheep_y_train, sheep_y_test = label_sheep[:num_of_train], label_sheep[num_of_train:(num_of_train + num_of_test)]

    # Turtle
    turtle_x_train, turtle_x_test = data_turtle[:num_of_train], data_turtle[num_of_train:(num_of_train + num_of_test)]
    turtle_y_train, turtle_y_test = label_turtle[:num_of_train], label_turtle[num_of_train:(num_of_train + num_of_test)]

    # Hedgehog
    hedgehog_x_train, hedgehog_x_test = data_hedgehog[:num_of_train], data_hedgehog[num_of_train:(num_of_train + num_of_test)]
    hedgehog_y_train, hedgehog_y_test = label_hedgehog[:num_of_train], label_hedgehog[num_of_train:(num_of_train + num_of_test)]

    # Octopus
    octopus_x_train, octopus_x_test = data_octopus[:num_of_train], data_octopus[num_of_train:(num_of_train + num_of_test)]
    octopus_y_train, octopus_y_test = label_octopus[:num_of_train], label_octopus[num_of_train:(num_of_train + num_of_test)]

    # Giraffe
    giraffe_x_train, giraffe_x_test = data_giraffe[:num_of_train], data_giraffe[num_of_train:(num_of_train + num_of_test)]
    giraffe_y_train, giraffe_y_test = label_giraffe[:num_of_train], label_giraffe[num_of_train:(num_of_train + num_of_test)]

    # Cat
    cat_x_train, cat_x_test = data_cat[:num_of_train], data_cat[num_of_train:(num_of_train + num_of_test)]
    cat_y_train, cat_y_test = label_cat[:num_of_train], label_cat[num_of_train:(num_of_train + num_of_test)]

    # Butterfly
    butterfly_x_train, butterfly_x_test = data_butterfly[:num_of_train], data_butterfly[num_of_train:(num_of_train + num_of_test)]
    butterfly_y_train, butterfly_y_test = label_butterfly[:num_of_train], label_butterfly[num_of_train:(num_of_train + num_of_test)]

    # Lion
    lion_x_train, lion_x_test = data_lion[:num_of_train], data_lion[num_of_train:(num_of_train + num_of_test)]
    lion_y_train, lion_y_test = label_lion[:num_of_train], label_lion[num_of_train:(num_of_train + num_of_test)]

    # fish
    fish_x_train, fish_x_test = data_fish[:num_of_train], data_fish[num_of_train:(num_of_train + num_of_test)]
    fish_y_train, fish_y_test = label_fish[:num_of_train], label_fish[num_of_train:(num_of_train + num_of_test)]

    # concatenate
    x_train = np.concatenate([bird_x_train[:], sheep_x_train[:], turtle_x_train[:], hedgehog_x_train[:], octopus_x_train[:], giraffe_x_train[:], cat_x_train[:], butterfly_x_train[:], lion_x_train[:], fish_x_train[:]])
    y_train = np.concatenate([bird_y_train[:], sheep_y_train[:], turtle_y_train[:], hedgehog_y_train[:], octopus_y_train[:], giraffe_y_train[:], cat_y_train[:], butterfly_y_train[:], lion_y_train[:], fish_y_train[:]])

    x_test = np.concatenate([bird_x_test[:], sheep_x_test[:], turtle_x_test[:], hedgehog_x_test[:], octopus_x_test[:], giraffe_x_test[:], cat_x_test[:], butterfly_x_test[:], lion_x_test[:], fish_x_test[:]])
    y_test = np.concatenate([bird_y_test[:], sheep_y_test[:], turtle_y_test[:],hedgehog_y_test[:], octopus_y_test[:], giraffe_y_test[:], cat_y_test[:], butterfly_y_test[:], lion_y_test[:], fish_y_test[:]])

    # Shuffle the order
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    for i in range(len(x_train)):
        x_train_reshape.append(np.reshape(x_train[i],(28,28,1)))

    for i in range(len(x_test)):
        x_test_reshape.append(np.reshape(x_test[i],(28,28,1)))

    # Normalizing the images
    x_train_norm = tf.keras.utils.normalize(x_train_reshape, axis=1)
    x_test_norm = tf.keras.utils.normalize(x_test_reshape, axis=1)  

    return x_train_reshape, y_train, x_test_reshape, y_test, x_train_norm, x_test_norm

def create_model():

    # Creating a simple model
    model = Sequential()

    model.add(Conv2D(32,(5,5), padding ='same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(5,5), padding ='same', activation='relu'))
    model.add(Conv2D(64,(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def plot_first_items():

    #Plot the first 25 entries accompanied with a label-prediction for each one
    plt.figure(figsize=(10,10))

    for i in range(25):

        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img1 = x_test[i]
        plt.imshow(img1.reshape((28,28)), cmap=plt.cm.binary)

        plt.ylabel(animals[y_test[i]].capitalize())
        plt.xlabel(animals[np.argmax(predictions[i])] + ' ' + str(round(100*np.max(predictions[i]), 2)) + '%')
        
    plt.show()

def plot_con_matrix():
    temp_list = []

    for i in range(len(y_test)):
        temp_list.append(np.argmax(predictions[[i]]))

    cm = confusion_matrix(y_test, temp_list)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(animals)),
            yticks=np.arange((len(animals))),
            xticklabels=animals, 
            yticklabels=animals,
            ylim=[-0.5, (len(animals) - 0.5)],
            title='Confusion Matrix',
            xlabel= 'Predicted animal',
            ylabel='True animal')

    #Rotate the x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()

def plot_evaluation_metrics():
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



########################################################################################
#################################### Main ##############################################
########################################################################################

animals = ['bird', 'sheep', 'turtle', 'hedgehog', 'octopus',
             'giraffe', 'cat', 'butterfly', 'lion', 'fish']

#return train, test vectors. Input: training items per object, test items per object
x_train, y_train, x_test, y_test, x_train_norm, x_test_norm = process_data(10000,5000)

#Create convolutional neural network
model = create_model()

#Tensorboard init
# val_loss = 0
# val_acc = 0
# tensorboard = TensorBoard(log_dir='logs\{}'.format(time()))
# cost = tf.summary.scalar("cost", val_loss)
# accuracy = tf.summary.scalar("accuracy", val_acc)
# train_summary_op = tf.summary.merge([cost,accuracy])

# train_writer = tf.summary.FileWriter(log_dir+'/train', graph=tf.get_default_graph())



# Training compilation and history
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_norm, y_train, epochs=3)

# Testing the model
val_loss, val_acc= model.evaluate(x_test_norm, y_test)
print(val_loss, val_acc)

# Predict (Predicts all images in test set and creates an array containing all predictions)
predictions = model.predict(x_test_norm)

#train_writer.add_summary(train_summary_str, step)

#tensorboard("logs/rmsprop")
# Save the model an loading it into the program again. Also saving it in js format
#model.save('doodle_model')
#new_model = tf.keras.models.load_model('doodle_model')
tfjs.converters.save_keras_model(model, "doodle_model_js")

# Plot first items in test with predicted and true values
plot_first_items()

# Plot convolutional matrix
plot_con_matrix()

# # Plot training & validation accuracy values
plot_evaluation_metrics()
