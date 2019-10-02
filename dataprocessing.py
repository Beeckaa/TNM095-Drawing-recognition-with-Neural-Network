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
label_bird = [0] * len(data_bird)
label_sheep = [1] * len(data_sheep)
label_turtle = [2] * len(data_turtle)

#Separating training data and testing data

#bird
bird_x_train, bird_x_test = data_bird[:40000], data_bird[40000:45000]
bird_y_train, bird_y_test = label_bird[:40000], label_bird[40000:45000]

#sheep
sheep_x_train, sheep_x_test = data_sheep[:40000], data_sheep[40000:45000]
sheep_y_train, sheep_y_test = label_sheep[:40000], label_sheep[40000:45000]

#turtle
turtle_x_train, turtle_x_test = data_turtle[:40000], data_turtle[40000:45000]
turtle_y_train, turtle_y_test = label_turtle[:40000], label_turtle[40000:45000]

# concatenate
x_train = np.concatenate([bird_x_train[:],sheep_x_train[:], turtle_x_train[:]])
y_train = np.concatenate([bird_y_train[:],sheep_y_train[:], turtle_y_train[:]])

x_test = np.concatenate([bird_x_test[:],sheep_x_test[:], turtle_x_test[:]])
y_test = np.concatenate([bird_y_test[:],sheep_y_test[:], turtle_y_test[:]])

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
# model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

# # Training
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)

# # Testing the model
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)

# # Save the model an loading it into the program again
# model.save('doodle_model')
new_model = tf.keras.models.load_model('doodle_model')

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
