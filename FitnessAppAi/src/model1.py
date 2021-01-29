import warnings
from datetime import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import Activation
import keras

warnings.filterwarnings("ignore")

# Constants 

IMG_W, IMG_H = 100, 100
NUM_OF_CLASSES = 131
BATCH_SIZE = 32
EPOCHS = 5

BASE_PATH = "D:/Fakultet/7sms-Soft_computing"
TRAIN_DIR = BASE_PATH + "/fruits-360/Training"
TEST_DIR = BASE_PATH + "/fruits-360/Test"
SAVE_DIR = "../saved_models/"

# start timer
start = datetime.now()

# prepare images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2)  # % used for validation set

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,  # same directory as training data
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    subset='validation')  # set as validation data

# Model


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(IMG_W, IMG_H, 3)))
model.add(Activation("relu"))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation("relu"))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_OF_CLASSES, activation="softmax"))

model.summary()

# Compile model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

fit_generator = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)
# using validation split

try:
    path = SAVE_DIR + 'model1.h5'

    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save(path)
except OSError as err:
    print("OS error: {0}".format(err))
except:
    print("Could not save model.")

# end timer
end = datetime.now()

duration = end - start
print("Duration: ", duration)


