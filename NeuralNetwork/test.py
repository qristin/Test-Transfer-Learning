import cv2
import utils
import math
import numpy as np
from skimage import exposure

s = "Price: $ %8.2f"% (356.08977)
print(s)

pathBowls = '../data/training-data/images/bowl/'
pathVases = '../data/training-data/images/vase/'

# take 20 images from each class
batchSize = 20

# for each image in folder generate hog feature and add class namen (0 or 1)
data = []
data = utils.getHogFeatures(pathBowls, 0, batchSize)
data += utils.getHogFeatures(pathVases, 1, batchSize)

# split data in train and test data
Data = utils.getTestAndTrainData(data, 7)

TrainData = Data.TrainData
TrainLabels = Data.TrainLabels
TestData = Data.TestData
TestLabels = Data.TestLabels

# Train the network with 2 classes (bowl/vase)
numberOfClasses = 2
model = utils.trainNetwork(TrainData, TrainLabels, TestData, TestLabels, numberOfClasses)

# -------------TESTING---------------------

# Testing the model with a BOWL image
bowlImagePath = pathBowls + '1f8f3014-e6a2-47bb-8b41-5d0fff021eba.png'
(hog, hogImage) = utils.getHog(bowlImagePath)

# show image of hog features of test image
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG image", hogImage)
print('Press any key to continue...')
cv2.waitKey(0)

# Predict class with trained model
input = hog.reshape(1,8100)
out = model.predict(input)
print("%5.2f%% that it belongs to class vase"% (out[0][0]*100))

# Testing the model with a VASE image
vaseImagePath = pathVases + '3a6e3748-cada-4457-bd44-6fab69330796.png'
(hog2, hogImage2) = utils.getHog(vaseImagePath)

# show image of hog features of test image
hogImage2 = exposure.rescale_intensity(hogImage2, out_range=(0, 255))
hogImage2 = hogImage2.astype("uint8")
cv2.imshow("HOG image", hogImage2)
print('Press any key to continue...')
cv2.waitKey(0)

# Predict class with trained model
input = hog2.reshape(1,8100)
out = model.predict(input)
print("%5.2f%% that it belongs to class vase"% (out[0][0]*100))
