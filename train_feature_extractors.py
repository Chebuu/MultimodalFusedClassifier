
import random
from multiprocessing import Pool
#workers = multiprocessing.cpu_count() - 1

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, Conv3D, MaxPool3D, UpSampling3D, ZeroPadding3D, GlobalMaxPool3D, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from numpy import random
import numpy as np
import os
from math import floor, ceil
# import matplotlib.pyplot as plt
from random import shuffle
import h5py

import subprocess

import shutil



from data_chunker import save_all_chunks, loadTrainingTestingFiles, loadTrainingTestingSets, loadChunksFromFileList, generate_chunks

# Here, we train each individual model to produce some high level features.
# The training and validation files are already shuffled.
def train_feature_extraction_models(training_files, validation_files, total_epochs=10):

    model_save_dir = "saved_model/fusion"
    if not os.path.exists(model_save_dir):
        subprocess.call('mkdir saved_model', shell=True)
        subprocess.call('mkdir ' + model_save_dir, shell=True)

    batch_size = 64 #  Of these chunks, we use this many as a batch


    modality_dirs = ["image_chunks", "audio_chunks", "imu_chunks", "audio_image_chunks"]
    saved_model_names = ["video_extractor", "audio_extractor", "imu_extractor", "audio_image_extractor"]
    tensorboard_names = ["video_extractor", "audio_extractor", "imu_extractor", "audio_image_extractor"]

    # trainModel(model_save_dir, training_files, validation_files, batch_size, modality_dir=modality_dirs[0], \
    # save_name=saved_model_names[0], tensorboard_folder=tensorboard_names[0])

    for chosen_index in range(0, 4):

        trainModel(model_save_dir, training_files, validation_files, batch_size, modality_dir=modality_dirs[chosen_index], \
        save_name=saved_model_names[chosen_index], tensorboard_folder=tensorboard_names[chosen_index], total_epochs=total_epochs)



def trainModel(model_save_dir, training_filelist, validation_files, batch_size, modality_dir, save_name, tensorboard_folder, total_epochs):

    print("Training on " + str(modality_dir))

    example_input,_ = loadChunksFromFileList(parent_dir = modality_dir, filelist=[training_filelist[0]])

    #print(example_input.shape)

    steps_per_epoch = ceil(len(training_filelist) / batch_size)
    validation_steps = ceil(len(validation_files) / batch_size)

    model = None
    input_shape = example_input.shape[1:]
    #print(input_shape)

    if modality_dir == "image_chunks":

        #n_samples, x_1, x_2, x_3, x_4 = example_input.shape
        #input_shape = (x_1, x_2, x_3, x_4)
        model = createVideoModel(input_shape, 6) #  We have 6 activities

    elif modality_dir == "audio_chunks":
        #x_1, x_2 = example_input.shape
        input_length = input_shape[0]
        model = createAudioModel(input_length, 6)

    elif modality_dir == "imu_chunks":
        #print(input_shape)
        input_length = input_shape[0] * input_shape[1]
        model = createIMUModel(input_length, 6)

    elif modality_dir == "audio_image_chunks":
        #print(input_shape)
        model = createAudioImageModel(input_shape, 6)


    if not os.path.exists(model_save_dir + "/" + save_name):
        subprocess.call('mkdir ' + model_save_dir + "/" + save_name, shell=True)

    save_model_callback = ModelCheckpoint(model_save_dir + "/" + save_name + "/epoch_{epoch:05d}.h5", period=1)

    model.fit_generator( generate_chunks(parent_dir=modality_dir, files=training_filelist, batch_size=batch_size),
                    epochs=total_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_data=generate_chunks(parent_dir=modality_dir, files=validation_files, batch_size=batch_size),#,callbacks=[TensorBoard(log_dir="logs/3dCNN")]
                    validation_steps =validation_steps,
                    callbacks=[TensorBoard(log_dir="logs/" + tensorboard_folder), save_model_callback])

    #CNN.save(model_save_dir + "/video_set_" + str(i) +  ".h5")


def createVideoModel(input_tuple, num_labels):

    print(input_tuple)

    inp = Input(shape=input_tuple)

    x = Conv3D(16, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(inp)
    x = Conv3D(16, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool3D(pool_size=(1,2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv3D(16, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(x)
    x = Conv3D(16, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(x)
    #x = MaxPool3D(pool_size=(2,2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv3D(8, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(x)
    x = Conv3D(8, kernel_size=(3,3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool3D(pool_size=(2,2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv3D(8, kernel_size=(1,3,3), activation=activations.relu, padding="valid")(x)
    x = Conv3D(8, kernel_size=(1,3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool3D(pool_size=(1,2,2))(x)
    x = Dropout(rate=0.2)(x)



    x = Conv3D(8, kernel_size=(1,3,3), activation=activations.relu, padding="valid")(x)
    x = Conv3D(8, kernel_size=(1,3,3), activation=activations.relu, padding="valid",name="features")(x)
    x = MaxPool3D(pool_size=(1,2,2))(x)
    x = Dropout(rate=0.2)(x)


    x = Flatten()(x)
    x = Dense(1024, activation=activations.relu)(x)
    out = Dense(num_labels, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())
    return model


def createAudioImageModel(input_tuple, num_labels):  #This creates the 1D CNN model

    print(input_tuple)
    inp = Input(shape=input_tuple)
    #inp = Input(shape=input_tuple)
    #inp = Input(shape=(input_length, 1))

    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(inp)
    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(inp)
    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(inp)
    x = Conv2D(32, kernel_size=(3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(16, kernel_size=(3,3), activation=activations.relu, padding="valid")(x)
    x = Conv2D(16, kernel_size=(3,3), activation=activations.relu, padding="valid")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Flatten()(x)
    dense_1 = Dense(64, activation=activations.relu, name="features")(x)
    dense_1 = Dense(num_labels, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())
    return model

def createAudioModel(input_length, num_labels):  #This creates the 1D CNN model

    print(input_length)

    #inp = Input(shape=input_tuple)
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

    dense_1 = Dense(128, activation=activations.relu, name="features")(img_1)
    dense_1 = Dense(32, activation=activations.relu)(dense_1)
    dense_1 = Dense(num_labels, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())
    return model

def createIMUModel(input_length, num_labels):  #This creates the 1D CNN model

    #inp = Input(shape=input_tuple)

    print(input_length)

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
    img_1 = GlobalMaxPool1D(name="features")(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    #print(img_1.shape)

    #dense_1 = Dense(64, activation=activations.relu)(img_1)
    #dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(num_labels, activation=activations.softmax)(img_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    #opt = optimizers.Adam(0.0001)
    opt = optimizers.Adam(lr = 0.01)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())
    return model

#train_feature_extraction_models()

#CODE FROM https://github.com/CVxTz/audio_classification/blob/master/code/keras_cnn_starter.ipynb
#TODO: I think the original audio was 48kHz, and you downsampled to 16kHz

    # Loop through entire file list and train models for each 'dataset' of total_dataset_size
    # for i in range(0, (num_total_files//total_dataset_size)+1):
    #
    #     print("Loading Dataset : " + str(i) + " of size " + str(total_dataset_size))
    #
    #     training_data, training_labels = None, None
    #     validation_data, validation_labels = None, None
    #
    #     training_file_subset = []
    #     validation_file_subset = []
    #
    #     # There are two cases - one is where we have enough data of training_size, and the other is if we have some small remainder left.
    #     if (i+1) * training_size < len(training_files):
    #         training_file_subset = training_files[i*training_size : (i+1) *training_size]
    #         training_data, training_labels = loadChunksFromFileList(parent_dir = "image_chunks", filelist=training_file_subset)
    #     else:
    #         training_file_subset = training_files[i*training_size :]
    #         training_data, training_labels = loadChunksFromFileList(parent_dir = "image_chunks", filelist=training_file_subset)
    #
    #     if (i+1) * validation_size < len(validation_files):
    #         validation_file_subset = validation_files[i*validation_size : (i+1) *validation_size]
    #         validation_data, validation_labels = loadChunksFromFileList(parent_dir = "image_chunks", filelist=validation_file_subset)
    #     else:
    #         validation_file_subset = validation_files[i*validation_size :]
    #         validation_data, validation_labels = loadChunksFromFileList(parent_dir = "image_chunks", filelist=validation_file_subset)
