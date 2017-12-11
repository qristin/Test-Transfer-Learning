import cv2
import math
import random
import collections

import numpy as np

from os import walk
from skimage import feature
from skimage import exposure
from keras.layers import Dense
from keras.models import Sequential

def getHogFeatures(imagePath, label, batchSize):
  # get pathes for all images in directory (imagePath)
  files = []
  for (dirpath, dirnames, filenames) in walk(imagePath):
    files.extend(filenames)
    break

  # get hog feature for each image
  hogFeatures = []
  for f in files[:batchSize]:
    (H, _) = getHog(imagePath + f)
    D = np.append(H, label)
    hogFeatures.append(D)

  return hogFeatures

def getHog(imagePath):
  # load image
  img = cv2.imread(imagePath,0)

  # get size of image
  height, width = img.shape[:2]

  (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
  hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
  hogImage = hogImage.astype("uint8")

  return H, hogImage

def getHogSize256(imagePath):
  # load image
  img = cv2.imread(imagePath,0)
  # resize image (256, 256)
  res = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
  (H, hogImage) = feature.hog(res, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
  hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
  hogImage = hogImage.astype("uint8")

  return H, hogImage

def getTestAndTrainData(dataArray, seed):
  if (seed):
    random.seed(seed)

  random.shuffle(dataArray)
  d = []
  labels = []
  for dataSlice in dataArray:
    d.append(dataSlice[:-1])
    labels.append(dataSlice[-1])
  size = math.floor(len(d)/2)
  data = collections.namedtuple('Data', ['TrainData', 'TrainLabels' , 'TestData', 'TestLabels'])(np.asarray(d[:size]), np.asarray(labels[:size]), np.asarray(d[size:]), np.asarray(labels[size:]))
  return data

def trainNetwork(train_data, train_labels, test_data, test_labels, numberOfClasses):
  num_hog_features = len(train_data[0])
  batch_size = len(train_data)
  np.transpose(train_data)
  np.transpose(test_data)

  # simple neural network with 1 hidden layer with 32 nodes
  model = Sequential()
  model.add(Dense(32, activation='relu',input_dim=num_hog_features))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  #compile network
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  #train network
  history = model.fit(train_data, train_labels, epochs=20, verbose=2, batch_size=batch_size, validation_data=(test_data, test_labels))

  #evaluate network
  [test_loss, test_acc] = model.evaluate(test_data, test_labels)
  print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
  return model

