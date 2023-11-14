#!/usr/bin/env python3

from os.path import expanduser
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths

from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader

# path to input dataset
DATASET = expanduser('~/dlcv/code/datasets/animals')

# get the list of image paths
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

# loop over multiple regularizers
for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the specified regularization function
    print(f'training model with {r} penalty')
    model = SGDClassifier(
        loss="log_loss",
        penalty=r,
        max_iter=100, # epochs
        learning_rate="constant",
        tol=1e-3,
        eta0=0.01,
        random_state=12
    )
    model.fit(features_train, labels_train)

    # evaluate the classifier
    acc = model.score(features_test, labels_test)
    print(f'{r} penalty accuracy: {acc*100:.2f}%')
