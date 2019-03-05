import gc
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from numpy import random
import librosa
import numpy as np
import glob
import os
import pandas as pd
#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def shuffleData():
    data_dir = "./sensor"

    raw_tuples = []  # List of tuples where each tuple is (audio_dir, label)

    if os.path.exists(data_dir):
        filepaths = os.listdir(data_dir)

        for f in filepaths:
            label = f.split("seq")[0].split("act")[1]
            label = int(label)
            raw_tuples.append( (data_dir + "/" + f, label) )

    np.random.shuffle(raw_tuples)  #Randomly shuffle the dataset
    print(raw_tuples)
    raw_files = [ tup[0] for tup in raw_tuples]  # Take the audio filenames
    raw_labels = [ tup[1] for tup in raw_tuples] #Take the corresponding audio labels

    return raw_files, raw_labels


num_rows = 150
num_cols = 19
input_length = num_rows * num_cols
def load_file(file_path, input_length=input_length):

    data = pd.read_csv(file_path, header=None, nrows=num_rows)
    data = data.values.flatten()  #Now we have a 1d array of each row in the csv
    #print(data)
    return data




def createModel(list_labels):  #This creates the 1D CNN model

    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Convolution1D(16, kernel_size=7, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=7, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = MaxPool1D(pool_size=4)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    #img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = MaxPool1D(pool_size=4)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    # img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    #print(img_1.shape)

    #dense_1 = Dense(64, activation=activations.relu)(img_1)
    #dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(img_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    #opt = optimizers.Adam(0.0001)
    opt = optimizers.Adam(lr = 0.01)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

batch_size = 5
def train_generator(list_files, file_to_label_dict, batch_size=batch_size):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_file(fpath) for fpath in batch_files]
            #print(batch_data)
            batch_data = np.array(batch_data)
            batch_data = np.expand_dims(batch_data, axis=2)
            if batch_data.shape != (batch_size, input_length, 1):
                continue
            #batch_data = batch_data.reshape(batch_size, input_length)
            #print(batch_data.shape)
            #print("WOW")
            batch_labels = [file_to_label_dict[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)
            yield batch_data, batch_labels




#train_files = glob.glob("../input/audio_train/*.wav")
#test_files = glob.glob("../input/audio_test/*.wav")
#train_labels = pd.read_csv("../input/train.csv")

inputFiles, labels = shuffleData()
print(labels)
model = createModel(labels)

#Create a mapping between the name of a file and the label of it
#file_to_label = { inputFiles[i] : labels[i] for i in np.arange(labels) }
file_to_label = {  inputFiles[i] : labels[i] for i in range(len(labels)) }


tr_files, val_files = train_test_split(inputFiles, test_size=0.1)
# data = load_file("./sensor/act01seq03.csv")
# print(data)
model.fit_generator(train_generator(tr_files, file_to_label), steps_per_epoch=len(tr_files)//batch_size, epochs=10,
                   validation_data=train_generator(val_files, file_to_label), validation_steps=len(val_files)//batch_size,
                  use_multiprocessing=True, workers=8, max_queue_size=20)


# TODO: You fixed incorrect batches in train_generator by removing incorrect shapes
#  Do try to find a better solution than this.
