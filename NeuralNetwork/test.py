import cv2
import utils
import math
import numpy as np
from skimage import exposure


pathBowls = '../data/training-data/images/bowl/'
pathVases = '../data/training-data/images/vase/'

# for each image in folder generate hog feature and add class namen (0 or 1)
data = []
data = utils.getHogFeatures(pathBowls, 0)
data += utils.getHogFeatures(pathVases, 1)

# split data in train and test data
Data = utils.getTestAndTrainData(data, 7)

TrainData = Data.TrainData
TrainLabels = Data.TrainLabels
TestData = Data.TestData
TestLabels = Data.TestLabels

# Train the network with 2 classes (bowl/vase)
numberOfClasses = 2
model = utils.trainNetwork(TrainData, TrainLabels, TestData, TestLabels, numberOfClasses)

# Testing the model with following image
testImagePath = pathBowls + '0f3dcca5-5909-45bd-9e25-637976fd32b4.png'
(hog, hogImage) = utils.getHog(testImagePath)

# show image of hog features of test image
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG image", hogImage)
print('Press any key to continue...')
cv2.waitKey(0)

# Predict class with trained model
# out = model.predict(hog)
# print(out)