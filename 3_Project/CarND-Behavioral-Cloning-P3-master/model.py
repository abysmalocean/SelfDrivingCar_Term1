import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
import tqdm
import pandas as pd
import os
import cv2


Center_image = 0
Left_image = 1
Right_iamge = 2
Steering_angle = 3
INPUT_SHAPE = (64, 64, 3)
learning_rate = 0.01
BATCH_SIZE = 128
EPOCHS = 5

def loadImage(path = "../data/driving_log.csv"):
    '''
    this fun is used for load the traning data from the tranning image folder and rade the csv file
    output is the trining files name and according steering value.
    '''
    # loand the csv file
    driving_data = pd.read_csv(path,header=None)
    #print(driving_data.describe())

    # assign x and y

    X = driving_data[[Center_image, Left_image, Right_iamge]].values
    y = driving_data[Steering_angle].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    return X_train, X_valid, y_train, y_valid


def argument(image, angle,top = 60, bottom = 25,size = INPUT_SHAPE):
    image = crop(image, top, bottom)
    image, angle = flip(image, angle)
    image = random_brightness(image)
    #image, angle = random_translate(image, angle, range_x = 100, range_y = 10)
    image = cv2.resize(image,(size[1],size[0]))
    return image, angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def crop(image, top = 60, bottom = 25):
    # Crop the image, remove the top sky and the front of the care
    return image[top:-bottom, :, :]

def flip(image, angle, prob=0.5):
    coin = np.random.rand()
    if coin < prob:
        return np.fliplr(image), -angle
    return image, angle

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def generator(X,y, batch_size=32, angle_argument = 0.22):

    num_samples = len(X)
    X_train = np.zeros([batch_size,INPUT_SHAPE[0],INPUT_SHAPE[1],INPUT_SHAPE[2]])
    y_train = np.zeros(batch_size)
    while 1: # Loop forever so the generator never terminates
        batch_i = 0
        for i in np.random.permutation(X.shape[0]):
            Center, Left, Right = X[i]
            angle = y[i]
            random_img = np.random.randint(0, 3)

            if random_img == Center_image:
                name = '../data/IMG/' + Center.split('/')[-1]
                angle = float(angle)
            elif random_img == Left_image:
                name = '../data/IMG/' + Left.split('/')[-1]
                angle = float(angle) + angle_argument
            elif random_img == Right_iamge:
                name = '../data/IMG/' + Right.split('/')[-1]
                angle = float(angle) - angle_argument

            image = cv2.imread(name)
            a_image,a_angle = argument(image, angle)
            X_train[batch_i] = a_image
            y_train[batch_i] = a_angle
            batch_i += 1
            if batch_i == batch_size:
                break

        yield (X_train, y_train)


def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

    model.add(Convolution2D(24, 5, 5,activation='relu', border_mode='same',subsample=(2, 2),name='conv1'))
    model.add(Convolution2D(36, 5, 5,activation='relu', border_mode='same',subsample=(2, 2),name='conv2'))
    model.add(Convolution2D(48, 5, 5,activation='relu', border_mode='same',subsample=(2, 2),name='conv3'))
    model.add(Convolution2D(64, 3, 3,activation='relu', border_mode='same',subsample=(1, 1),name='conv4'))
    model.add(Convolution2D(64, 3, 3,activation='relu', border_mode='same',subsample=(1, 1),name='conv5'))
    model.add(Dropout(0.50))
    model.add(Flatten(name='flatten'))
    #model.add(Dense(1164, activation='relu',name='Dense1'))
    model.add(Dense(100 , activation='relu',name='Dense2'))
    model.add(Dense(50  , activation='relu',name='Dense3'))
    model.add(Dense(10  , activation='relu',name='Dense4'))
    model.add(Dense(1   , activation='relu',name='Dense5'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    return model

def train(model,X_train, X_valid, y_train, y_valid):

    model.fit_generator(generator(X_train,y_train, batch_size = BATCH_SIZE, angle_argument = 0.22),
                        samples_per_epoch = 500 * (X_train.shape[0] // BATCH_SIZE),
                        nb_epoch=EPOCHS,
                        validation_data = generator(X_valid,y_valid, batch_size = BATCH_SIZE, angle_argument = 0.22),
                        nb_val_samples = len(100 * X_valid),
                        callbacks=[ModelCheckpoint('tranning.h5',verbose=0,monitor='val_loss', save_best_only=False)],
                        verbose=1)
    model.save_weights("model.h5", overwrite=True)

X_train, X_valid, y_train, y_valid = loadImage()
model =model()
#model.summary()
train(model,X_train, X_valid, y_train, y_valid)
print("Liang Xu")
