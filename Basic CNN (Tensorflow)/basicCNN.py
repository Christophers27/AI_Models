# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Load data
xTrain = np.loadtxt('Data/input.csv', delimiter=',')
yTrain = np.loadtxt('Data/labels.csv', delimiter=',')

xTest = np.loadtxt('Data/input_test.csv', delimiter=',')
yTest = np.loadtxt('Data/labels_test.csv', delimiter=',')

# Reshape data
xTrain = xTrain.reshape(len(xTrain), 100, 100, 3)
yTrain = yTrain.reshape(len(yTrain), 1)

xTest = xTest.reshape(len(xTest), 100, 100, 3)
yTest = yTest.reshape(len(yTest), 1)

# All the values in xTrain and xTest are RGB values between 0 and 255, so we
# divide them by 255 to normalize them
xTrain = xTrain / 255
xTest = xTest / 255

# Print shape
print("Shape of xTrain: ", xTrain.shape)
print("Shape of yTrain: ", yTrain.shape)
print("Shape of xTest: ", xTest.shape)
print("Shape of yTest: ", yTest.shape)

# Show a random sample
i = random.randint(0, len(xTrain))
plt.imshow(xTrain[i, :])
plt.show()
