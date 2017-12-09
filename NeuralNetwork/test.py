import utils
import math
import numpy as np

pathBowls = '../data/training-data/images/bowl/'
pathVases = '../data/training-data/images/vase/'

data = []
data = utils.getHogFeatures(pathBowls, 0)
data += utils.getHogFeatures(pathVases, 1)

Data = utils.getTestAndTrainData(data, 7)

TrainData = Data.TrainData
TrainLabels = Data.TrainLabels

TestData = Data.TestData
TestLabels = Data.TestLabels

# 2 classes (bowl/vase)
numberOfClasses = 2

model = utils.trainNetwork(TrainData, TrainLabels, TestData, TestLabels, numberOfClasses)

testImagePath = pathBowls + '0f3dcca5-5909-45bd-9e25-637976fd32b4.png'
hog = utils.getHog(testImagePath)

out = model.predict(hog)

print(out)