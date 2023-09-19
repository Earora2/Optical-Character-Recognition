# Optical-Character-Recognition
Hindi Character Recognition using KNN

The system uses the dataset ‘dataset_ka_kha’. The images will be loaded into the system. They will be converted to greyscale and reshaped to (28,28) size.
Then, the image data will be taken in the form of an array of size 784. This data is loaded into the KNN model, using the following specifications:

N neighbours = 7
Leaf size = 30
Metric = Minkowski
Weights = uniform

This achieved an accuracy of 93.46%.
