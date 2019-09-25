import numpy as np

# Load data (numpy 28x28 bitmaps)
data_bird = np.load('bitmaps/bird.npy')
data_sheep = np.load('bitmaps/sheep.npy')
data_turtle = np.load('bitmaps/seaturtle.npy')


# Save npy-file as a binary file
str = data_bird.tostring()
print(len(str))

# Number of unique doodles (Subtract 80 header bytes in the array that describe the data)
total = (len(str)-80) / (28*28)
print(total)

start = 80