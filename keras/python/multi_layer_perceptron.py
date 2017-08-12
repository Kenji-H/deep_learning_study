import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

RESHAPED = 28 * 28
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2

NB_EPOCH = 20
BATCH_SIZE = 128
OPTIMIZER = SGD()
N_HIDDEN = 128

VERBOSE = 1

# for reproducibility
np.random.seed(1226)

# download data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten data
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# build model
model = Sequential()

# 1st hidden layer
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))

# 2nd hidden layer
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))

# output layer
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

# fit model
history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evaluate model
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test loss:{}".format(score[0]))      # 0.186985414364
print('Test accuracy:{}'.format(score[1]))  # 0.9469
