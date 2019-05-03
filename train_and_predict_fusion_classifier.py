
import random
from multiprocessing import Pool
#workers = multiprocessing.cpu_count() - 1

from keras import optimizers, losses, activations, models, Model
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
from itertools import permutations


from data_utils import loadArrayFromFile, saveArrayToFile, saveDictionaryToFile

def load_data_and_label(parent_dir, filename):

    data = loadArrayFromFile(parent_dir + "/" + filename)
    label = int( filename[filename.index("act_") + 4 : filename.index("_index")] )
    data = data.flatten()

    return data, label

def batch_load_feature_files(parent_dir, filelist):

    batch_data = []
    batch_labels = []

    for filename in filelist:

        data, label = load_data_and_label(parent_dir, filename)

        batch_data.append(data)
        batch_labels.append(label)

    batch_data = np.vstack(batch_data)
    batch_labels = np.vstack(batch_labels)

    return batch_data, batch_labels

def generate_chunks(parent_dir, files, batch_size):

    while True:

        batch_files = random.choice(a= files, size=batch_size)
        batch_data, batch_labels = batch_load_feature_files(parent_dir = parent_dir, filelist=batch_files)
        #batch_data = np.expand_dims(batch_data, axis=2)
        #print(batch_data.shape)
        #print(batch_labels.shape)
        yield( batch_data, batch_labels )

def train_feature_fusion_model(training_dir, validation_dir, model_save_path = "saved_model/fusion/fusion_model", total_epochs=10):

    if not os.path.exists(model_save_path):
        subprocess.call('mkdir ' + model_save_path, shell=True)

    training_filelist = os.listdir(training_dir)
    validation_files = os.listdir(validation_dir)

    batch_size = 64 #  Of these chunks, we use this many as a batch

    example_input = loadArrayFromFile(training_dir + "/" + training_filelist[0])  #Get some example input
    input_length = example_input.shape[-1:]

    steps_per_epoch = ceil(len(training_filelist) / batch_size)
    validation_steps = ceil(len(validation_files) / batch_size)

    model = FusionClassifier(input_length, 6)
    tensorboard_folder = "fusion_model"


    save_model_callback = ModelCheckpoint(model_save_path + "/epoch_{epoch:05d}.h5", period=1)

    model.fit_generator( generate_chunks(parent_dir=training_dir, files=training_filelist, batch_size=batch_size),
                    epochs=total_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_data=generate_chunks(parent_dir=validation_dir, files=validation_files, batch_size=batch_size),#,callbacks=[TensorBoard(log_dir="logs/3dCNN")]
                    validation_steps =validation_steps,
                    callbacks=[TensorBoard(log_dir="logs/" + tensorboard_folder), save_model_callback])

# Convert a feature file to an id - this is used to calculate the confidence of a sensor modality combination for an activity
# An id is of the form of a string XYYY where X is the activity label and YYY is the sensor modality combination
#  Also calculates the confidence for a sample of data.
def computeConfidenceForSample(scores, filename):

    activity_str = filename[filename.index("act_") + 4 : filename.index("_index")]
    activity = int(activity_str)
    combo_str = filename[filename.index("combo_") + 6 : filename.index(".")]
    id = activity_str + combo_str

    confidence = scores[activity] - np.max(np.delete(scores, activity))
    sample_confidence = max(0, confidence)

    return confidence, id, activity

#  Run prediction on the feature fusion classifier using the augmented set, producing scores (MAKE SURE YOU SAVE THE SCORES, NOT THE NORMALIZED OUTPUT)
#    i.e. this should produce files that are like act_X_index_X_combo_010.features and act_X_index_X_combo_010.scores
#    Use the scores to measure the confidence of the model
#   Once prediction is complete, save the confidences to a file.
def classify_dataset(model_path, feature_dir = "features", dataset_dir="augmented_training", score_save_dir="scores", statistics_dir="statistics", isTesting=False):

    if not os.path.exists(score_save_dir + "/" + dataset_dir):
        subprocess.call('mkdir ' + score_save_dir, shell=True)
        subprocess.call('mkdir ' + score_save_dir + "/" + dataset_dir, shell=True)

    confidence_dictionary = {}

    model = load_model(model_path)

    prediction_files = os.listdir(feature_dir + "/" + dataset_dir)

    prior1_output = 'prior1'
    prior2_output = 'prior2'

    total_correct = 0
    total_incorrect = 0

    print("Classifying Dataset under " + dataset_dir)

    #intermediate_layer_model_1 = Model(inputs=model.input, outputs=model.get_layer(prior1_output).output)
    #intermediate_layer_model_2 = Model(inputs=model.input, outputs=model.get_layer(prior1_output).output)

    # Run prediction on each file, and save the scores to the correct folder.
    for i in range(0, len(prediction_files)):
    #for i in range(0, 5):

        file_to_predict = prediction_files[i]
        input_data, input_label = load_data_and_label(feature_dir + "/" + dataset_dir, file_to_predict)


        input_data = input_data.reshape(1, input_data.shape[0])
        #print(input_data.shape)
        output = model.predict(input_data, batch_size=1)
        output = output.flatten()

        predicted_activity = np.argmax(output)

        # Compute the confidence and id for this sample
        confidence, id, activity = computeConfidenceForSample(output, file_to_predict)

        if id not in confidence_dictionary:
            confidence_dictionary[id] = [confidence, 1]
        else:
            confidence_dictionary[id][0] += confidence
            confidence_dictionary[id][1] += 1

        save_filename = file_to_predict[: file_to_predict.index(".features")] + ".scores"
        save_filepath = score_save_dir + "/" + dataset_dir + "/" + save_filename
        saveArrayToFile(output, save_filepath)

        # Check if your predictions are correct:
        if predicted_activity == activity:
            total_correct += 1
        else:
            total_incorrect += 1

    if isTesting:
        print("Fusion Classifier Test: ")
        print(" \t num correct: " + str(total_correct) + " / incorrect: " + str(total_incorrect))

    if not os.path.exists(statistics_dir):
        subprocess.call('mkdir ' + statistics_dir, shell=True)
    saveDictionaryToFile(confidence_dictionary, statistics_dir + "/model_confidences_" + dataset_dir + ".pkl")


def FusionClassifier(input_tuple, num_classes):


    print(input_tuple)

    inp = Input(shape=(input_tuple))

    x = Dense(128, activation=activations.relu)(inp)

    x = Dense(64, activation=activations.relu)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(32, activation=activations.relu, name="prior1")(x)

    out = Dense(num_classes, activation=activations.softmax, name="prior2")(x)
    #out = Dense(num_classes, activation=activations.softmax, name="prior2")(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())
    return model
