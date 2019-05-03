
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

from data_utils import loadArrayFromFile, saveDictionaryToFile


def load_data_from_file(filepath, isScore=False):

    data = loadArrayFromFile(filepath)
    data = data.flatten()

    # If this is a score file, we need to append some modailty list information
    if isScore:
        modality_list = convert_scorepath_to_combolist(filepath)
        data = np.append(data, modality_list)

    #data = np.expand_dims(data, axis=0)

    return data

def batch_load_files(filepath_list, isScore=False):

    batch_data = []

    for filepath in filepath_list:

        data = load_data_from_file(filepath, isScore)

        batch_data.append(data)

    batch_data = np.vstack(batch_data)

    return batch_data

def generate_chunks(source_files, target_files, batch_size):

    while True:

        random_indexes = random.choice(np.arange(len(source_files)), batch_size )
        random_source_files = [source_files[i] for i in random_indexes]
        random_target_files = [target_files[i] for i in random_indexes]

        batch_input = batch_load_files(filepath_list = random_source_files, isScore=True)
        batch_output = batch_load_files(filepath_list = random_target_files)

        #print(batch_input.shape)
        #print(batch_output.shape)

        yield( batch_input, batch_output )

#from data_chunker import save_all_chunks, loadTrainingTestingFiles, loadTrainingTestingSets, loadChunksFromFileList, generate_chunks

# Here, we train each individual model to produce some high level features.
# The training and validation files are already shuffled.
def train_feature_generator_models(model_save_dir, score_dir, feature_dir, modality_dirs, total_epochs=10):

    print("Training Feature Generator Models")

    if not os.path.exists(model_save_dir):
        subprocess.call('mkdir saved_model', shell=True)
        subprocess.call('mkdir ' + model_save_dir, shell=True)

    batch_size = 64 #  Of these chunks, we use this many as a batch


    training_files = os.listdir(score_dir + "/augmented_training")
    validation_files = os.listdir(score_dir + "/augmented_validation")


    #modality_feature_dirs = ["image_chunks", "imu_chunks", "audio_image_chunks"]
    # saved_model_names = ["video_extractor", "audio_extractor", "imu_extractor", "audio_image_extractor"]
    # tensorboard_names = ["video_extractor", "audio_extractor", "imu_extractor", "audio_image_extractor"]

    # trainModel(model_save_dir, training_files, validation_files, batch_size, modality_dir=modality_dirs[0], \
    # save_name=saved_model_names[0], tensorboard_folder=tensorboard_names[0])

    for i in range(0, len(modality_dirs)):
    #for modality in in modality_dirs:
        modality = modality_dirs[i]
        save_name = modality + "_generator"

        training_scorepaths, training_featurepaths = convert_scorepaths_to_featurepaths(training_files, feature_dir, \
        "training", modality_dirs, i)
        training_scorepaths = ["scores/augmented_training/" + x for x in training_scorepaths]

        validation_scorepaths, validation_featurepaths = convert_scorepaths_to_featurepaths(validation_files, feature_dir, \
        "validation", modality_dirs, i)
        validation_scorepaths = ["scores/augmented_validation/" + x for x in validation_scorepaths]

        #training_files = [feature_dir + "/training/" + modality + "/" + f for f in training_files]
        #validation_files = [feature_dir + "/validation/" + modality + "/" + f for f in training_files]

        trainModel(feature_dir, model_save_dir, training_scorepaths, training_featurepaths, \
        validation_scorepaths, validation_featurepaths, batch_size, modality_index = i, modality_list=modality_dirs, \
        save_name=save_name, tensorboard_folder=save_name, total_epochs=total_epochs)



# Ok, so when we get a filename, it is going to be of the form "act_0_index_11_combo_010.scores.npy"
#   This is the input to the model.
#  We need to convert it of the form "training/modality/act_0_index_22.features.npy"
#   This is the output of the model.
#  Basically, this function takes a list of score paths, figures out which ones are actually relevant for this modality,
#   Returns the list of relevant score paths, and the corresponding feature paths.
def convert_scorepaths_to_featurepaths(scorepaths, feature_dir, dataset_dir, modality_list, modality_index):

    final_scorepaths = []
    final_featurepaths = []

    for scorepath in scorepaths:

        combo_str = scorepath[ scorepath.index("combo_") + 6 : scorepath.index(".") ]
        featurefilename = scorepath[ : scorepath.index("_combo")] + ".features.npy"

        # this modality is present in this combo, and is therefore relevant.
        if int(combo_str[modality_index]) > 0:

            final_scorepaths.append(scorepath)
            featurefilepath = feature_dir + "/" + dataset_dir + "/" + modality_list[modality_index] + "/" + featurefilename
            final_featurepaths.append(featurefilepath)

    return final_scorepaths, final_featurepaths


# Input is of format "act_0_index_11_combo_010.scores.npy"
def convert_scorepath_to_combolist(scorepath):
    combo_str = scorepath[ scorepath.index("combo_") + 6 : scorepath.index(".") ]
    combolist = [int(x) for x in combo_str]

    return combolist


def trainModel(feature_dir, model_save_dir, training_scorepaths, training_featurepaths, validation_scorepaths, validation_featurepaths,\
 batch_size, modality_index, modality_list, save_name, tensorboard_folder, total_epochs):

    print("Training on " + str(modality_list[modality_index]))

    #example_input,_ = loadChunksFromFileList(parent_dir = modality_dir, filelist=[training_filelist[0]])
    example_input = load_data_from_file(training_scorepaths[0], isScore=True)
    input_shape = example_input.shape
    example_output = load_data_from_file(training_featurepaths[0])
    output_shape = example_output.shape

    print(input_shape)
    print(output_shape)

    steps_per_epoch = ceil(len(training_scorepaths) / batch_size)
    validation_steps = ceil(len(validation_scorepaths) / batch_size)

    model = None
    #input_shape = example_input.shape[1:]
    #print(input_shape)

    modality_dir = modality_list[modality_index]
    if modality_dir == "image_chunks":

        #n_samples, x_1, x_2, x_3, x_4 = example_input.shape
        #input_shape = (x_1, x_2, x_3, x_4)
        model = createVideoModel(input_shape, output_shape)

    elif modality_dir == "audio_chunks":
        #x_1, x_2 = example_input.shape
        #input_length = input_shape[0]
        model = createAudioModel(input_shape, output_shape)

    elif modality_dir == "imu_chunks":
        #print(input_shape)
        #input_length = input_shape[0] * input_shape[1]
        model = createIMUModel(input_shape, output_shape)

    elif modality_dir == "audio_image_chunks":
        #print(input_shape)
        model = createAudioImageModel(input_shape, output_shape)


    if not os.path.exists(model_save_dir + "/" + save_name):
        subprocess.call('mkdir ' + model_save_dir + "/" + save_name, shell=True)

    save_model_callback = ModelCheckpoint(model_save_dir + "/" + save_name + "/epoch_{epoch:05d}.h5", period=1)

    model.fit_generator( generate_chunks(source_files=training_scorepaths, target_files=training_featurepaths, batch_size=batch_size),
                    epochs=total_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_data=generate_chunks(source_files=validation_scorepaths, target_files=validation_featurepaths, batch_size=batch_size),#,callbacks=[TensorBoard(log_dir="logs/3dCNN")]
                    validation_steps =validation_steps,
                    callbacks=[TensorBoard(log_dir="logs/" + tensorboard_folder), save_model_callback])


def createVideoModel(input_tuple, output_tuple):

    print(input_tuple)
    print(output_tuple[0])

    inp = Input(shape=input_tuple)

    x = Dense(32, activation=activations.softmax)(inp)
    x = Dense(64, activation=activations.softmax)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(128, activation=activations.softmax)(x)
    out = Dense(output_tuple[0], activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001)

    #model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc']) #produces accuracy of around 79%
    model.compile(optimizer=opt, loss=losses.mean_absolute_error, metrics=['mae'])
    print(model.summary())
    return model


def createAudioImageModel(input_tuple, output_tuple):  #This creates the 1D CNN model

    print(input_tuple)

    inp = Input(shape=input_tuple)

    x = Dense(16, activation=activations.softmax)(inp)
    x = Dense(32, activation=activations.softmax)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(56, activation=activations.softmax)(x)
    out = Dense(output_tuple[0], activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001)

    #model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.compile(optimizer=opt, loss=losses.mean_absolute_error, metrics=['mae'])
    print(model.summary())
    return model

def createAudioModel(input_tuple, output_tuple):  #This creates the 1D CNN model

    print(input_tuple)

    inp = Input(shape=input_tuple)

    x = Dense(32, activation=activations.softmax)(inp)
    x = Dense(64, activation=activations.softmax)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(128, activation=activations.softmax)(x)
    out = Dense(output_tuple[0], activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.001)

    #model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.compile(optimizer=opt, loss=losses.mean_absolute_error, metrics=['mae'])
    print(model.summary())
    return model

def createIMUModel(input_tuple, output_tuple):  #This creates the 1D CNN model

    #inp = Input(shape=input_tuple)

    print(input_tuple)

    inp = Input(shape=input_tuple)

    x = Dense(16, activation=activations.softmax)(inp)
    x = Dropout(rate=0.2)(x)
    x = Dense(24, activation=activations.softmax)(x)
    #x = Dense(32, activation=activations.softmax)(x)
    out = Dense(output_tuple[0], activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    #opt = optimizers.Adam(0.0001)
    opt = optimizers.Adam(lr = 0.01)

    #model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.compile(optimizer=opt, loss=losses.mean_absolute_error, metrics=['mae'])
    print(model.summary())
    return model

# scorepath name is of the format ".../act_0_index_11_combo_010.scores.npy"
def getActivityFromScorepath(scorepath):

    activity_str = scorepath[scorepath.index("act_") + 4 : scorepath.index("_index")]
    return int(activity_str)

# Run classification on a specific feature generator.
#  Returns the penalty for activities for this modality.
def classifyGenerator(model_path, scorepaths, featurepaths):

    print("Classifying " + str(model_path))

    model = load_model(model_path)

    activity_dict = {} # Of the format activity : [original_feature_average, generated_feature_average, num_datapoints]

    # Need two vectors, one for each modality, otherwise only one gets updated with all additions.
    zeroed_features_orig = np.zeros_like(load_data_from_file(featurepaths[0]))
    zeroed_features_gen = np.zeros_like(load_data_from_file(featurepaths[0]))

    total_datapoints = len(scorepaths)

    #Iterate through the training_scorepaths and calculate the feature penalty
    for i in range(0, len(scorepaths)):
        source_file = scorepaths[i]
        target_file = featurepaths[i]
        activity = getActivityFromScorepath(source_file)

        input_data = load_data_from_file(source_file, isScore=True)
        true_output = load_data_from_file(target_file).flatten()

        #print(true_output.shape)

        input_data = input_data.reshape(1, input_data.shape[0])
        #print(input_data.shape)
        output = model.predict(input_data, batch_size=1)
        output = output.flatten()

        #print("Generated: " + str(output[20:26]))
        #print("True: " + str(true_output[20:26]))

        #print(output.shape)

        if activity not in activity_dict:
            activity_dict[activity] = [zeroed_features_orig, zeroed_features_gen, 1]
        else:
            activity_dict[activity][0] += true_output
            activity_dict[activity][1] += output
            activity_dict[activity][2] += 1

            #print("Generated: " + str(output[20:26]))
            #print("True: " + str(true_output[20:26]))
            # print(activity_dict[activity][0][20:26])
            # print(activity_dict[activity][1][20:26])
            #
            # break


    penalty_dict = {} # similar to activity_dict, but items are just the penalties.

    for key, item in activity_dict.items():

        average_generated_feature = item[1]/item[2]
        average_original_feature = item[0]/item[2]

        #print(average_generated_feature[:5])
        #print(average_original_feature[:5])

        penalty = np.linalg.norm(average_generated_feature - average_original_feature)# / np.linalg.norm(average_generated_feature)
        penalty_dict[key] = penalty

    #print(activity_dict)

    # print(activity_dict[0][0][:5])  #Original Features
    # print(activity_dict[0][1][:5])  #Generated Feature

    return penalty_dict


#  Assumes original dictionary, dict_to_add are of format activity : penalties
def addDictionaryPenalties(original_dictionary, dict_to_add):

    # If original dictionary is empty, return the dict_to_add.
    if not original_dictionary:
        return dict_to_add

    new_dictionary = {}
    for key, value in dict_to_add.items():
        new_dictionary[key] = original_dictionary[key] + value

    return new_dictionary

#  Run prediction on the feature generators given the previous score files and corresponding feature files
#   to get the average distance of generated features as well as calculating the average distance of original features
#  Once prediction is complete, save the penalties to a file (Don't incorporate the hyperparameter lambda just yet)
def classifyGenerators(model_save_dir, score_dir, feature_dir, modality_dirs, statistics_dir, total_epochs):

    #batch_size = 64 #  Of these chunks, we use this many as a batch


    training_files = os.listdir(score_dir + "/augmented_training")
    validation_files = os.listdir(score_dir + "/augmented_validation")

    # This is a dictionary containing keys of modality_dirs, and the items are the dictionaries of activity : penalty
    modality_penalty_dict_training = {}
    modality_penality_dict_validation = {}
    total_activity_penalty_training = {}  # This is a dictionary containing the total penalty for an activity across all modalities.
    total_activity_penalty_validation = {}

    for i in range(0, len(modality_dirs)):
    #for modality in in modality_dirs:
        modality = modality_dirs[i]
        save_name = modality + "_generator"  # This is the model name.
        model_path = model_save_dir + "/" + save_name + "/epoch_000" + str(total_epochs) + ".h5"

        print("Converting Scorepaths for " + str(modality))
        training_scorepaths, training_featurepaths = convert_scorepaths_to_featurepaths(training_files, feature_dir, \
        "training", modality_dirs, i)
        training_scorepaths = ["scores/augmented_training/" + x for x in training_scorepaths]

        validation_scorepaths, validation_featurepaths = convert_scorepaths_to_featurepaths(validation_files, feature_dir, \
        "validation", modality_dirs, i)
        validation_scorepaths = ["scores/augmented_validation/" + x for x in validation_scorepaths]

        #training_files = [feature_dir + "/training/" + modality + "/" + f for f in training_files]
        #validation_files = [feature_dir + "/validation/" + modality + "/" + f for f in training_files]

        penalty_dict_training = classifyGenerator(model_path, training_scorepaths, training_featurepaths)
        modality_penalty_dict_training[modality] = penalty_dict_training
        total_activity_penalty_training = addDictionaryPenalties(total_activity_penalty_training, penalty_dict_training)

        penalty_dict_validation = classifyGenerator(model_path, validation_scorepaths, validation_featurepaths)
        modality_penality_dict_validation[modality] = penalty_dict_validation
        total_activity_penalty_validation = addDictionaryPenalties(total_activity_penalty_validation, penalty_dict_validation)

    # First iterate through the total penalty counts
    # Then normalize the feature generator penalties according to the activity (basically, normalize for activity 1 across all sensor modalities)
    # keys are the modality, items are the dictionary items where each activity is a key and the non-normalized penalty is the item.
    for activity, _ in total_activity_penalty_training.items():

        for key, value in modality_penalty_dict_training.items():
            modality_penalty_dict_training[key][activity] /= total_activity_penalty_training[activity]

        for key, value in modality_penality_dict_validation.items():
            modality_penality_dict_validation[key][activity] /= total_activity_penalty_validation[activity]

    # print(modality_penalty_dict_training)
    # print(modality_penality_dict_validation)
    print("Saving Training and Validation Penalties.")
    saveDictionaryToFile(modality_penalty_dict_training, statistics_dir + "/penalties_training.pkl")
    saveDictionaryToFile(modality_penality_dict_validation, statistics_dir + "/penalties_validation.pkl")
