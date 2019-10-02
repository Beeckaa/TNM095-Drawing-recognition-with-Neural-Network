import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import tensorflowjs as tfjs

# Load data (numpy 28x28 bitmaps)
data_bird = np.load('bitmaps/bird.npy')
data_sheep = np.load('bitmaps/sheep.npy')
data_turtle = np.load('bitmaps/seaturtle.npy')
data_hedgehog = np.load('bitmaps/hedgehog.npy')
data_octopus = np.load('bitmaps/octopus.npy')
data_giraffe = np.load('bitmaps/giraffe.npy')

# Create label arrays
label_bird = [0] * len(data_bird)
label_sheep = [1] * len(data_sheep)
label_turtle = [2] * len(data_turtle)
label_hedgehog = [3] * len(data_hedgehog)
label_octopus = [4] * len(data_octopus)
label_giraffe = [5] * len(data_giraffe)

# Separating training data and testing data

# Bird
bird_x_train, bird_x_test = data_bird[:40000], data_bird[40000:45000]
bird_y_train, bird_y_test = label_bird[:40000], label_bird[40000:45000]

# Sheep
sheep_x_train, sheep_x_test = data_sheep[:40000], data_sheep[40000:45000]
sheep_y_train, sheep_y_test = label_sheep[:40000], label_sheep[40000:45000]

# Turtle
turtle_x_train, turtle_x_test = data_turtle[:40000], data_turtle[40000:45000]
turtle_y_train, turtle_y_test = label_turtle[:40000], label_turtle[40000:45000]

# Hedgehog
hedgehog_x_train, hedgehog_x_test = data_hedgehog[:40000], data_hedgehog[40000:45000]
hedgehog_y_train, hedgehog_y_test = label_hedgehog[:40000], label_hedgehog[40000:45000]

# Octopus
octopus_x_train, octopus_x_test = data_octopus[:40000], data_octopus[40000:45000]
octopus_y_train, octopus_y_test = label_octopus[:40000], label_octopus[40000:45000]

# Giraffe
giraffe_x_train, giraffe_x_test = data_giraffe[:40000], data_giraffe[40000:45000]
giraffe_y_train, giraffe_y_test = label_giraffe[:40000], label_giraffe[40000:45000]

# concatenate
x_train = np.concatenate([bird_x_train[:], sheep_x_train[:], turtle_x_train[:], hedgehog_x_train[:], octopus_x_train[:], giraffe_x_train[:]])
y_train = np.concatenate([bird_y_train[:], sheep_y_train[:], turtle_y_train[:], hedgehog_y_train[:], octopus_y_train[:], giraffe_y_train[:]])

x_test = np.concatenate([bird_x_test[:], sheep_x_test[:], turtle_x_test[:], hedgehog_x_test[:], octopus_x_test[:], giraffe_x_test[:]])
y_test = np.concatenate([bird_y_test[:], sheep_y_test[:], turtle_y_test[:], hedgehog_y_test[:], octopus_y_test[:], giraffe_y_test[:]])

# Shuffle the order
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

# Normalizing the images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# # Creating a simple model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
<<<<<<< HEAD
# model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
=======
# model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))
>>>>>>> 8e2d775f4617c6155cf68bb9b66bb192a5ad29cb

# # Training
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=2)

# # Testing the model
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)

# # Save the model an loading it into the program again. Also saving it in js format
# model.save('doodle_model')
new_model = tf.keras.models.load_model('doodle_model')
tfjs.converters.save_keras_model(new_model, "doodle_model_js")

# Predict (Predicts all images in x_test and creates an array with each prediction)
predictions = new_model.predict(x_test)

########################################################################################
#################################### Plotting ##########################################
########################################################################################


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
    #plt.ylabel(y_test[i])
    plt.ylabel(round(100*np.max(predictions[i]), 2))
plt.show()


########################################################################################
################################## Confusion Matrix ###################################
########################################################################################



#Now let's visualize the errors between the predictions 
#and the actual labels using a confusion matrix
from sklearn.metrics import confusion_matrix

# plt.xlabel('Predicted Values')
# plt.title('Correct Values')

temp_list = []

for i in range(len(y_test)):
    if np.argmax(predictions[[i]]) == 0:
        temp_list.append(0)
        
    elif np.argmax(predictions[[i]]) == 1:
        temp_list.append(1)
       
    elif np.argmax(predictions[[i]]) == 2:
        temp_list.append(2)
        

cm = confusion_matrix(y_test, temp_list)

# cm.plt.xlabel('Predicted Values')
# cm.plt.title('Correct Values')
plt.matshow(cm)
plt.show()
