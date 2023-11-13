#!/usr/bin/env python3

from os.path import expanduser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader

# path to input dataset
DATASET = expanduser('~/dlcv/code/datasets/animals')

# number of nearest neighbors for classification
NEIGHBORS = 1

# number of jobs for k-NN distance (-1 uses all available cores)
NUM_JOBS = -1

# get the list of images in the dataset
image_path_list = list(paths.list_images(DATASET))
if not image_path_list:
    print('no files found')
    quit()

# initialize the image preprocessor
sp = SimplePreprocessor(width=32, height=32)

# load the dataset from disk
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_path_list, verbose=500)
print(data.shape)

# reshape the data matrix
num_files, height, width, channels = data.shape
print(f'{num_files=}')
print(f'{height=}')
print(f'{width=}')
print(f'{channels=}')
data = data.reshape((num_files, height * width * channels))
print(data.shape)

# encode the labels as integers (from text)
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the data into training (75%) and testing (25%) sets
(features_train, features_test, labels_train, labels_test) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# train a k-NN classifier on the raw pixel intensities
model = KNeighborsClassifier(n_neighbors=NEIGHBORS, n_jobs=NUM_JOBS)
model.fit(features_train, labels_train)

# evaluate the k-NN classifier
print(classification_report(labels_test, # labels true
                            model.predict(features_test), # labels predicted
                            target_names=le.classes_))
