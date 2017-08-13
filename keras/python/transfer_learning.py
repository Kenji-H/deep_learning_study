from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

NB_CLASSES = 10
NB_SAMPLE = 1000

NB_EPOCH = 20
BATCH_SIZE = 40
OPTIMIZER = Adam()

VERBOSE = 1

# for reproducibility
np.random.seed(1226)

# download data (just use subset of data due to data size)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train[:NB_SAMPLE]
y_train = y_train[:NB_SAMPLE]
X_test = X_test[:NB_SAMPLE]
y_test = y_test[:NB_SAMPLE]

# enlarge images to fit in InceptionV3 format
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = np.repeat(np.repeat(X_train, 8, axis=1), 8, axis=2)
X_test = np.repeat(np.repeat(X_test, 8, axis=1), 8, axis=2)

# normalize data
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# one-hot encoding
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# load base model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add custom part to pre-trained model
custom_model = base_model.output
custom_model = GlobalAveragePooling2D()(custom_model)
custom_model = Dense(500, activation='relu')(custom_model)
custom_model = Dense(10, activation='softmax')(custom_model)
model = Model(input=base_model.input, output=custom_model)

# freeze InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=["accuracy"])

# data augmentation
data_generator = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest')

# train
model.fit_generator(
    data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=NB_EPOCH, steps_per_epoch=NB_SAMPLE/BATCH_SIZE, verbose=VERBOSE)

# evaluate
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test loss:{}".format(score[0]))      # 0.973047031403
print('Test accuracy:{}'.format(score[1]))  # 0.664

