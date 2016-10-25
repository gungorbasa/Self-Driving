import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import csv
import decimal
import numpy as np
from PIL import Image


ch, row, col = 3, 192, 256
epoch = 32

# Reads labels and returns as yield, so ideal for big data
def read_labels(angles_path, size):
    trainY = []
    with open(angles_path, 'rb') as csvReader:
        for angle in csvReader:
            trainY.append(decimal.Decimal(angle))
    return trainY

# Reads images and returns as yield, so ideal for big data
def read_images(image_path, start, size):
    imgs = sorted(os.listdir(image_path))
    num = size
    j = 0
    data = np.empty((size,ch,row,col),dtype="float32")
    for i in xrange(start, start + size):
        if not imgs[i].startswith('.'):
            if image_path.endswith('/') != True:
                image_path += '/'
            img = Image.open(image_path + imgs[i])
            arr = np.asarray (img, dtype ="float32")
            data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]]
            j += 1
    return data
    

# Reads angles and images and returns them
def readData(image_path, angles_path, size=1000):
    start = 0
    labels = read_labels(angles_path, size)
    total_len = len(labels)
    while 1:
        if start + size > total_len:
            size = total_len - size
        data = read_images(image_path, start, size)
        lbls = labels[start:start+size]
        
        yield (data, lbls)

def create_model():
    model = Sequential()
    # Regulerize data
    model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

    


if __name__ == "__main__":
    image_path = './ImagesSDC'
    angles_path = './angles.csv'


    # for d, a in readData(image_path, angles_path):
    #     for i in d:
    #         print(d)
    model = create_model()
    # # print(trainX.shape, len(trainY), valX.shape, len(valY))

    model.fit_generator(readData(image_path, angles_path),
        nb_epoch=epoch,
        samples_per_epoch=1000,
        verbose=2
    )

    print("Saving model weights and configuration file.")
    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

