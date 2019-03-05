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
    audio_dir = "./sound"

    raw_audio_tuples = []  # List of tuples where each tuple is (audio_dir, label)

    if os.path.exists(audio_dir):
        label_dirs = os.listdir(audio_dir)

        for label in label_dirs:
            filenames = os.listdir(audio_dir + "/" + label)  #files in the current label vid_directory

            for f in filenames:
                raw_audio_tuples.append( (audio_dir + "/" + label + "/" + f , label) )

    np.random.shuffle(raw_audio_tuples)  #Randomly shuffle the dataset
    print(raw_audio_tuples)
    raw_audio_files = [ tup[0] for tup in raw_audio_tuples]  # Take the audio filenames
    raw_audio_labels = [ int(tup[1].split("act")[1]) for tup in raw_audio_tuples] #Take the corresponding audio labels

    return raw_audio_files, raw_audio_labels

def audio_norm(data):

    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

input_length = 16000
def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
    if len(data)>input_length:


        max_offset = len(data)-input_length

        offset = np.random.randint(max_offset)

        data = data[offset:(input_length+offset)]


    else:

        max_offset = input_length - len(data)

        offset = np.random.randint(max_offset)


        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")


    data = audio_norm(data)
    #print(data.shape)
    return data




def createModel(list_labels):  #This creates the 1D CNN model

    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu)(img_1)
    dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=0.001)

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
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:,:,np.newaxis]
            #print(batch_data.shape)
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

model.fit_generator(train_generator(tr_files, file_to_label), steps_per_epoch=len(tr_files)//batch_size, epochs=2,
                    validation_data=train_generator(val_files, file_to_label), validation_steps=len(val_files)//batch_size,
                   use_multiprocessing=True, workers=8, max_queue_size=20)



#CODE FROM https://github.com/CVxTz/audio_classification/blob/master/code/keras_cnn_starter.ipynb
#TODO: I think the original audio was 48kHz, and you downsampled to 16kHz
