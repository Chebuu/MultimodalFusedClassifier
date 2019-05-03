import h5py
from keras.models import load_model
from keras import Model

import os
import subprocess

import shutil
import numpy as np

from data_chunker import loadChunkAndLabelFromFile, createAugmentedDataCombinations
from data_utils import saveArrayToFile, loadArrayFromFile

#  Produce the features (i.e. run prediction) for training/validation/test data, creating DIFFERENT FOLDERS FOR ALL (make sure you can indicate what activity)
#     the filename can be something like act_X_index_X.features

# Run prediction on the models given the input files and take the output from the feature layer.
def produce_features(modality_dirs, model_paths, feature_save_dir, training_files, validation_files, testing_files, total_epochs):

    # Create the directory in which you will save all the features.
    if not os.path.exists(feature_save_dir):
        subprocess.call('mkdir ' + feature_save_dir, shell=True)

    #modality_dirs = ["image_chunks", "imu_chunks", "audio_image_chunks", "audio_chunks"]
    datasets = [training_files, validation_files, testing_files]


    # We iterate through each dataset type
    for i in range(0, len(datasets)):

        # Make the dataset training directory
        dataset_feature_save_dir = "features/"
        if i == 0:
            dataset_feature_save_dir += "training"
        elif i == 1:
            dataset_feature_save_dir += "validation"
        elif i == 2:
            dataset_feature_save_dir += "testing"

        if not os.path.exists(dataset_feature_save_dir):
            subprocess.call('mkdir ' + dataset_feature_save_dir, shell=True)


        # For each modality, we are going to run inference
        for j in range(0, len(modality_dirs)):

            predict_and_save_features(modality_dirs[j], model_paths[j], dataset_feature_save_dir, datasets[i])


# Run prediction and save the output from the "features" layer - MAKE SURE YOU FLATTEN THIS.
def predict_and_save_features(modality_dir, model_path, feature_save_dir, prediction_files):

    feature_parent_dir = feature_save_dir + "/" + modality_dir
    if not os.path.exists(feature_parent_dir):
        subprocess.call('mkdir ' + feature_parent_dir, shell=True)

    model = load_model(model_path)

    layer_name = 'features'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Run prediction on each file, and save the feature to the correct folder.
    for i in range(0, len(prediction_files)):

        chunk_file = prediction_files[i]
        input_data, input_label = loadChunkAndLabelFromFile(modality_dir + "/" + chunk_file)

        if modality_dir == "imu_chunks":
            input_data = input_data.flatten()
            input_data = input_data.reshape(1, input_data.shape[0], 1)
            #print(input_data.shape)

        if modality_dir == "audio_chunks":
            input_data = input_data.flatten()
            input_data = input_data.reshape(1, input_data.shape[0], 1)
        #print(input_data.shape)
        intermediate_output = intermediate_layer_model.predict(input_data, batch_size=1)
        feature_to_save = intermediate_output.flatten()  # Flatten the output

        save_filename = "act_" + str(input_label) + "_index_" + str(i) + ".features"
        save_filepath = feature_parent_dir + "/" + save_filename
        print("Saving feature of " + str(feature_to_save.shape) + " to " + str(save_filepath))
        saveArrayToFile(feature_to_save, save_filepath)

# produce an augmented set of features via combinations for each modality.
def createAugmentedCombo(permutation_list, modality_feature_files, new_feature_top_dir):

    # Get the numpy arrays for each modality
    modality_features_data = []
    for feature_file in modality_feature_files:
        #data_to_append = loadArrayFromFile(feature_file)
        #print(feature_file)
        #print(data_to_append.shape)
        modality_features_data.append(loadArrayFromFile(feature_file))
    modality_zeroed_features_data = []
    for feature_data in modality_features_data:
        modality_zeroed_features_data.append(np.zeros_like(feature_data))

    # Now we have two lists - one contains data, the other contains the same sized data but of all zeroes.
    # We can now use our combinations.

    for permutation in permutation_list:

        #print(permutation)

        current_data_combination = []

        for i in range(0, len(permutation)):
            decision_boolean = permutation[i]  #This will either be true or false.

            if decision_boolean: #We add
                current_data_combination.append(modality_features_data[i])
            else:
                current_data_combination.append(modality_zeroed_features_data[i])


        # Create a concatenated vector:
        current_data_combination = np.hstack(current_data_combination)
        #print(current_data_combination.shape)

        # Create the name for this combined feature vector.
        data_filename = modality_feature_files[0][modality_feature_files[0].index("ks/")+3 : modality_feature_files[0].index(".features")]
        data_filename += "_combo_"  + ''.join(str(int(e)) for e in permutation) + ".features"

        # Save the concatenated data to the file.
        saveArrayToFile(current_data_combination, new_feature_top_dir + "/" + data_filename)


# Produce the augmented dataset for a dataset (training, validation, test)
#  Produce an augmented set of features for those features (make sure you can tell what combination of modalities it uses)
#  i.e. something like act_X_index_X_combo_010.features
def produceAugmentedDataset(modality_dirs, new_dataset_dir = "augmented_training", dataset_dir = "training"):

    feature_top_dir = "features/" + dataset_dir
    new_feature_top_dir = "features/" + new_dataset_dir
    if not os.path.exists(new_feature_top_dir):
        subprocess.call('mkdir ' + new_feature_top_dir, shell=True)


    # This is a list of combinations for the modalities being available/not available.
    #  it will be something like [0,1], [1,0], [1,1]
    #print(modality_dirs)
    permutation_list = createAugmentedDataCombinations(modalities = modality_dirs)
    #print(permutation_list)

    example_feature_dir = feature_top_dir + "/" + modality_dirs[0]
    feature_filenames = os.listdir(example_feature_dir)

    # We iterate through each feature file, and take the data for each modality to produce an augmented dataset.
    for feature_filename in feature_filenames:

        modality_feature_files = []  # This is a list containing a set of features for each modality for the same chunk of data

        for modality_dir in modality_dirs:

            # This is the feature file for a modality.
            feature_file = feature_top_dir + "/" + modality_dir + "/" + feature_filename
            modality_feature_files.append(feature_file)

        createAugmentedCombo(permutation_list, modality_feature_files, new_feature_top_dir)
