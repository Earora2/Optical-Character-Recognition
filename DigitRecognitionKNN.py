# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 03:05:38 2019

@author: Eshaan
"""

from sklearn import datasets, svm, metrics
import scipy
import os


directory = '/testing'
output_directory = '/testing'

PRINT_SLICES = False
THRESHOLD_PIXELS_COUNT = 60000
MAX_BOUNDING_BOX_WIDTH = 675
MAX_BOUNDING_BOX_HEIGHT = 50

def detect_left_edge(image):
    h,w = image.shape
    max = 0
    edge = 0
    for x in range(0,100):

        vertical_slice = image[0:h,  x:x+15 ]
        vertical_slice_pixels_count = vertical_slice.sum()

        if( vertical_slice_pixels_count > THRESHOLD_PIXELS_COUNT):
            scipy.misc.imsave(output_directory + '/' + filename, image[0:h, x:675])
            return 0

        if (vertical_slice_pixels_count > max):
            max = vertical_slice_pixels_count
            edge = x * 2
    return edge

# loop over images
for filename in os.listdir(directory):
    if filename.endswith("negate.jpg"):
        input_image = scipy.misc.imread(directory + '/' + filename)
        image_height, image_width = input_image.shape
        max = 0;
        output_image = input_image
        # detect top edge of the image bounding box
        for h in range(0, image_height - MAX_BOUNDING_BOX_HEIGHT):
            temp_image = input_image[h:h + MAX_BOUNDING_BOX_HEIGHT, 0:0 + MAX_BOUNDING_BOX_WIDTH]
            if temp_image.sum() > max:
                max= temp_image.sum()
                output_image = temp_image
        edge = detect_left_edge(output_image)
        
from skimage.feature import hog
df= hog(training_digit_image, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))

import numpy as np
import os
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

features_list = []
features_label = []
# load labeled training / test data
# loop over the 10 directories where each directory stores the images of a digit
for digit in range(0,10):
    label = digit
    training_directory = '/traning_data_set/' + str(label) + '/'
    for filename in os.listdir(training_directory):
        if (filename.endswith('.png')):
            training_digit_image = scipy.misc.imread(training_directory + filename)
            training_digit = color.rgb2gray(training_digit)

            # extra digit's Histogram of Gradients (HOG). Divide the image into 5x5 blocks and where block in 10x10
            # pixels
       
            features_list.append(df)
            features_label.append(label)
            
# store features array into a numpy array
features  = np.array(features_list, 'float64')
# split the labled dataset into training / test sets
X_train, X_test, y_train, y_test = train_test_split(features, features_label)
# train using K-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# get the model accuracy
model_score = knn.score(X_test, y_test)

# save trained model
joblib.dump(knn, '/models/knn_model.pkl')


knn = joblib.load('/models/knn_model.pkl')
def feature_extraction(image):
    return hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(10, 10), cells_per_block=(5, 5))
def predict(df):
    predict = knn.predict(df.reshape(1,-1))[0]
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    return predict, predict_proba[0][predict]
digits = []
# load your image from file
# extract featuress
hogs = list(map(lambda x: feature_extraction(x), digits))
# apply k-NN model created in previous
predictions = list(map(lambda x: predict(x), hogs))

print(predictions)