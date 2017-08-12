import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np

NB_CLASSES = 10
IMG_ROWS = 28
IMG_COLS = 28
VALIDATION_SPLIT = 0.2

NB_EPOCH = 20
BATCH_SIZE = 128
OPTIMIZER = Adam()

VERBOSE = 1

# use Theano dimension ordering (channel, width, height)
keras.backend.set_image_dim_ordering('th')

# for reproducibility
np.random.seed(1226)

# download data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# add channel dimension to data.
# after this transformation, the dimension is (data_cnt, channel_cnt, row_size, col_size)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# one-hot encoding
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# build model
model = Sequential()

# create 20 feature maps(channel)
model.add(Convolution2D(20, kernel_size=5, padding="same",
                        input_shape=(1, IMG_ROWS, IMG_COLS)))
model.add(Activation("relu"))

# summarize sub matrix of feature maps
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# create 50 feature maps(channel)
model.add(Convolution2D(50, kernel_size=5, border_mode="same"))
model.add(Activation("relu"))

# summarize sub matrix of feature maps
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# flatten
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# output layer
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
              metrics=["accuracy"])
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evaluate model
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test loss:{}".format(score[0]))      # 0.0325818518888
print('Test accuracy:{}'.format(score[1]))  # 0.9913
