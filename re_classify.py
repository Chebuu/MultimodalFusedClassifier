



import random
from multiprocessing import Pool
#workers = multiprocessing.cpu_count() - 1

import os
from math import floor, ceil
# import matplotlib.pyplot as plt
from random import shuffle
import h5py
import numpy as np

import subprocess
from pprint import pprint
import matplotlib.pyplot as plt

import shutil

from keras.models import load_model

from data_utils import loadDictionaryFromFile, loadArrayFromFile, saveArrayToFile
from data_chunker import save_all_chunks, loadTrainingTestingFiles, loadTrainingTestingSets, loadChunksFromFileList, generate_chunks, createAugmentedDataCombinations
from train_feature_extractors import train_feature_extraction_models
from test_feature_extractors import produce_features, produceAugmentedDataset
from train_and_predict_fusion_classifier import train_feature_fusion_model, classify_dataset
from train_and_predict_feature_generators import train_feature_generator_models, classifyGenerators


#GENERAL ALGORITHM
# TRAINING:
#  Train the feature extraction models on some data, leaving out some test input data
#  Produce the features (i.e. run prediction) for training/validation/test data, creating DIFFERENT FOLDERS FOR ALL (make sure you can indicate what activity)
#     the filename can be something like act_X_index_X.features
#  Produce an augmented set of features for those features (make sure you can tell what combination of modalities it uses)
#  Train the feature fusion classifier on that augmented set
#  Run prediction on the feature fusion classifier using the augmented set, producing scores (MAKE SURE YOU SAVE THE SCORES, NOT THE NORMALIZED OUTPUT)
#    i.e. this should produce files that are like act_X_index_X_combo_010.features and act_X_index_X_combo_010.scores
#    Use the scores to measure the confidence of the model
#    Once prediction is complete, save the confidences to a file.
#  Train the feature generators given the score, available modalities, and corresponding features.
#  Run prediction on the feature generators given the previous score files and corresponding feature files
#   to get the average distance of generated features as well as calculating the average distance of original features
#  Once prediction is complete, save the penalties to a file (Don't incorporate the hyperparameter lambda just yet)
#
def train():

    split_ratio = [0.6,0.2,0.2]
    total_epochs = 50


    training_files, validation_files, testing_files = loadTrainingTestingFiles(parent_dir="image_chunks", split_ratio=split_ratio)

    # Shuffle the data
    shuffle(training_files)
    shuffle(validation_files)
    shuffle(testing_files)

    # Train the feature extraction models if we don't have the models already
    if not os.path.exists("saved_model/fusion"):
        train_feature_extraction_models(training_files, validation_files, total_epochs)


    # Remember - All feature data, when concatenated, is in this order: ["image_chunks", "imu_chunks", "audio_image_chunks"]
    #  This is determined here.
    #modality_dirs = ["image_chunks", "imu_chunks", "audio_image_chunks"]
    modality_dirs = ["image_chunks", "imu_chunks", "audio_chunks"]
    dataset_names = ["training", "validation", "testing"]
    new_dataset_names = ["augmented_training", "augmented_validation", "augmented_testing"]


    # Produce the features for training/validation/test data.
    feature_save_dir = "features"
    if not os.path.exists(feature_save_dir):

        # total epochs is used to find the latest trained model
        model_paths = ["saved_model/fusion/video_extractor/epoch_000" + str(total_epochs) +".h5",\
         "saved_model/fusion/imu_extractor/epoch_000" + str(total_epochs) +".h5", \
        "saved_model/fusion/audio_extractor/epoch_000" + str(total_epochs) +".h5"]  # These are h5 models that we want to run inference with
        produce_features(modality_dirs, model_paths, feature_save_dir, training_files, validation_files, testing_files, total_epochs)

        # Produce the augmented dataset
        for i in range(0, len(dataset_names)):
            produceAugmentedDataset(modality_dirs, new_dataset_dir =new_dataset_names[i], dataset_dir=dataset_names[i])

    # Train the feature fusion model
    fusion_model_save_path = "saved_model/fusion/fusion_model"
    if not os.path.exists(fusion_model_save_path):
        train_feature_fusion_model("features/augmented_training", "features/augmented_validation", fusion_model_save_path, total_epochs)


    statistics_dir = "statistics"
    fusion_scores_save_dir = "scores"
    if not os.path.exists(fusion_scores_save_dir):
        # Iterate through the training and validation dataset to produce the scores.
        for i in range(0, len(new_dataset_names)-1):

            # Do classification and save the scores.
            classify_dataset(fusion_model_save_path + "/epoch_000" + str(total_epochs) + ".h5", dataset_dir=new_dataset_names[i], \
            score_save_dir = fusion_scores_save_dir, statistics_dir=statistics_dir)


    #print(loadDictionaryFromFile("scores/model_confidences_augmented_training.pkl"))
    model_save_dir = "saved_model/generator"
    if not os.path.exists(model_save_dir):
        train_feature_generator_models(model_save_dir = model_save_dir, score_dir = "scores", feature_dir="features",\
         modality_dirs=modality_dirs, total_epochs=total_epochs)
        classifyGenerators(model_save_dir = model_save_dir, score_dir = "scores", feature_dir="features", \
        modality_dirs=modality_dirs, statistics_dir=statistics_dir, total_epochs=total_epochs)



    # TODO: Switch the epochs for the models back to 50 - this means for training, as well as the prediction for producing features.
    #         as well as for the feature fusion model, and for the feature generators (both traniing and classification)
    #  ALSO: Note that the accuracy is kinda too good - you might want to try using the raw audio data instead of the audio image
    #    Also look into why binary_crossentropy does way better than categorical crossentropy for feature generators.
    #     I'm concerned that it's just rounding teh desired features to 0,1.  Recheck the feature generators for IMU -it behaves much differently
    #    than the audio or video.
    #  In addition, for the classifyGenerators you didn't check the validation set, and you also didn't save the metrics to file.


# PREDICTION and EVALUATION:
#  Now use the test input data from the generated features under the test  (Features are already produced during training)
#   As before, produce an augmented set of features from these test features. (This is done during training)
#  Run prediction through the fusion network using these augmented features.
#     i.e. you have knowledge of:
#          - what combinations of sensor modalities are present
#          - the normalized scores of the classifier
#          - what activity this is classified as (prediction)
#          - what activity this should be classified as (ground truth)
#
#     Use the ModelContribution calculation to determine if you should fuse/how you should fuse
#     If you ought to fuse, also record the following:
#          - what the highest scoring ModelContribution is
#          - using the recommended sensor combinations, run classification: what are the new scores?
#          - what is the new activity classification (prediction)?
#
#    If you don't fuse, just save the results to another folder (nocontribute) - you can use this to calculate
#          - How many of these results are incorrect? (i.e. indicates how many the ModelContribution should have reconsidered)
#               - Of these incorrect results, how many were in each modality category and activity?
#    If you do fuse, save the results to another folder (contribute) and calculate
#          - How many of these results were originally correct, and then changed to be incorrect? (i.e. bad model contribution)
#          - How many of these results were original incorrect, and then changed to be correct?  (i.e. correct model contribution)
#               - Of both of these results, how many were in each modality category and activity?
#
#    You might want to run the prediction through several values of lambda (hyperparameter for discount)
#    You can also try to do a confusion matrix for each one.

def prediction(lam):

    feature_save_dir = "features"
    dataset_dir = "testing"
    total_epochs = 50 # Use to get the latest trained model for feature fusion
    num_activities = 6  #Total number of activities - used in calcuting the model contribution.

    possible_reclassification_count = 0

    #lam = 0.35  # Lambda that you are testing with.  The higher the lambda, the less the model reclassifies.
    # Essentially, you are trying to maximize the ratio of correct reclassifications to number of reclassifications.

    fusion_model_save_path = "saved_model/fusion/fusion_model"
    statistics_dir = "statistics"

    # Create the scores from the augmented testing features.
    if not os.path.exists("scores/augmented_testing"):
        classify_dataset(fusion_model_save_path + "/epoch_000" + str(total_epochs) + ".h5", dataset_dir="augmented_testing", \
        score_save_dir = "scores", statistics_dir=statistics_dir, isTesting=True)

    # Create the folder for the new scores generated by reclassification.
    #  Keep in mind that this folder will have <= the number of scores as the original augmented_testing.
    #  Basically, it only saves a score here if it has reclassified.
    reclassified_scores_dir = "scores/reclassified_testing"
    if not os.path.exists(reclassified_scores_dir):
        possible_reclassification_count = do_reclassification(reclassified_scores_dir, statistics_dir, total_epochs, num_activities, lam=lam)

    return possible_reclassification_count

# In addition to the regular evaluation of how many correct/incorrect etc,
#  You should also make an evaluation of what categories these are coming from.
#  i.e. for a correct classification to incorrect classification,
#  You can also compute the following:
#    1. how many of these instances were of a specific activity category?
#       Format:  { activity : count }
#
#    2. how many of these instances involved a specific modality at the beginning for an activity?
#      you can use this to compute the total instances involving this modality at the beginning
#       Format: {modality: [act1_count, act2_count, ...] ... }
#
#    3. how many of these instances involved adding a specific modality at the end for an activity?
#      you can use this to compute the total instances involving adding this modality at the end
#       Format: {modality: [act1_count, act2_count, ...]  ... }
#
#    3 and 4 can be combined into one dictionary, but I don't really feel like having to manage
#     a complicated structure i.e. intializing different components
#
#    4. how many of these instances involved a single modality, two modalities, ... for an activity
#      you can use this to figure out how many of these instances just involved a modality count
#       Format: {modality_count : [act1_count, act2_count, ...] }
#    5. how many of these instances involved adding a single modality, two modalities, ... for an activity
#      you can use this to figure out how many of these instances just involved a modality count
#       Format: {modality_count : [act1_count, act2_count, ...] }
#
#  //ignore
#    how many of these instances were of some old modality combination for some activity?
#    how many of these instances were of some new modality combination for some activity?
#    how many of these instances were by adding some modality set for some activity?
#  //end ignore
#

# This class handles adding these metrics to the dictionary
#  dictionary_list contains the dictionaries in order from the comments above.
def addToDictionaries(dictionary_list, true_activity, old_modality_list, added_modality_list, modality_names, num_classes=6):

    new_dictionary_list = []  # REMEMBER TO DO EACH ONE IN ORDER
    activity_add = [0 for i in range(0, num_classes)]
    activity_add[true_activity] = 1


    # Add to the activity count (#1)
    new_dictionary_list.append(addToDictionary(dictionary_list[0], true_activity, 1))

    #  Add to the modality starts with activities #2
    dictToAdd = dictionary_list[1].copy()
    for i in range(0, len(old_modality_list)):
        key = modality_names[i]
        if old_modality_list[i] > 0:  #This modality was used in this combination
            dictToAdd = addToDictionary( dictToAdd, key, activity_add).copy()
    new_dictionary_list.append( dictToAdd )

    # Add to the modality appended and activities #3
    dictToAdd = dictionary_list[2].copy()


    for i in range(0, len(added_modality_list)):
        key = modality_names[i]
        if added_modality_list[i] > 0:  #This modality was used in this combination
            dictToAdd = addToDictionary( dictToAdd, key, activity_add).copy()
    new_dictionary_list.append( dictToAdd )


    # Add to the modalities that used 1,2,3 modalities in their classification.
    num_modalities = sum(old_modality_list)
    new_dictionary_list.append( addToDictionary(dictionary_list[3], num_modalities, activity_add ).copy()  )

    # Add to the modailties that added 1,2.. modalities in their classification
    num_modalities = sum(added_modality_list)
    new_dictionary_list.append( addToDictionary(dictionary_list[4], num_modalities, activity_add ).copy()  )


    return new_dictionary_list


# Handles the intialization and cloning for adding to a dictionary.
#  Either you will set value here, or you will add value to the existing value in the dictionary.
def addToDictionary(dictionary, key, value):
    new_dictionary = dictionary.copy()

    if key not in new_dictionary:
        if isinstance(value, list):
            new_value = value.copy()
            new_dictionary[key] = new_value
        else:
            new_dictionary[key] = value

    else:
        # If the item to add is a list, we have to elementwise addition
        if isinstance(value, list):
            new_value = value.copy()
            for i in range(0, len(new_value)):
                new_dictionary[key][i] += new_value[i]
        else:
            new_dictionary[key] += value

    return new_dictionary

# Remember in the classification, we produced a bunch of newcombos (potentially more than the original)
#  This is because we also evaluated every modality set being added.
#  Just remember that we produced a bunch of reclassifications and most of them may be simply
#   adding different modality combos to the same original combo.

# One question we are asking is basically, given a combination of extra modalities, which ones
#  do we choose to combine with the old classifier?
#  We changed our ModelContribution to produce lists of contributions for each modality
#    This way, we can also evaluate whether or not the choices being made are effective.

def evaluate():

    # Files in this folder are of teh format: act_0_index_5_combo_010.scores.npy
    old_score_dir = "scores/augmented_testing"
    # Files in this folder are of the format: act_0_index_5_combo_010_newcombo_111.scores.npy
    new_score_dir = "scores/reclassified_testing"

    new_score_files = os.listdir(new_score_dir)
    old_score_files = os.listdir(old_score_dir)

    print("Num Classifications: " + str(len(old_score_files)))

    modality_dirs = ["image", "imu", "audio"]  # MAKE SURE THIS MATCHES WITH THE TRAINING AND PREDICTION


    # These are the metrics for reclassification
    num_incorrect_to_correct = 0
    num_correct_to_incorrect = 0
    num_incorrect_to_incorrect = 0
    num_correct_to_correct = 0

    correct_to_incorrect_dicts = [{} for i in range(0, 5)]
    incorrect_to_correct_dicts = [{} for i in range(0, 5)]

    #These are the metrics for no reclassification
    num_correct = 0
    num_incorrect = 0

    # Iterate through each new score, and compare with the old score.
    # For every new score, remove the corresponding old score from the file list.
    for new_score_file in new_score_files:

        # Get the correct activity
        true_prediction = int(new_score_file[ new_score_file.index("act_") + 4 : new_score_file.index("_index") ])

        # Find the corresponding old score filename
        old_score_filename = new_score_file[: new_score_file.index("_newcombo") ] + ".scores.npy"

        # Remove the old score from the file list.
        if old_score_filename in old_score_files:
            old_score_files.remove(old_score_filename)

        # Find the filepaths for the old and new score files, and get the numpy arrays.
        new_score_filepath = new_score_dir + "/" + new_score_file
        old_score_filepath = old_score_dir + "/" + old_score_filename
        new_scores = loadArrayFromFile(new_score_filepath)
        old_scores = loadArrayFromFile(old_score_filepath)

        old_modality_list = new_score_file[new_score_file.index("combo_") + 6 : new_score_file.index("_newcombo") ]
        old_modality_list = [int(c) for c in old_modality_list ]
        new_modality_list = new_score_file[new_score_file.index("newcombo_") + 9 : new_score_file.index(".scores") ]
        new_modality_list = [int(c) for c in new_modality_list]

        # print(new_score_file)
        # print(old_modality_list)
        # print(new_modality_list)

        added_modality_list = [new_modality_list[i] - old_modality_list[i] for i in range(0, len(old_modality_list))]

        # Get the top activity from both:
        top_generated_prediction = np.argmax(new_scores)
        top_old_prediction = np.argmax(old_scores)

        # Old prediction was wrong, new prediction is right
        if top_old_prediction != true_prediction and top_generated_prediction == true_prediction:
            num_incorrect_to_correct += 1

            incorrect_to_correct_dicts = addToDictionaries(incorrect_to_correct_dicts, true_prediction, \
            old_modality_list, added_modality_list, modality_dirs, num_classes=6).copy()

        # Old prediction was right, new prediction is wrong
        elif top_old_prediction == true_prediction and top_generated_prediction != true_prediction:
            num_correct_to_incorrect += 1

            correct_to_incorrect_dicts = addToDictionaries(correct_to_incorrect_dicts, true_prediction, \
            old_modality_list, added_modality_list, modality_dirs, num_classes=6)

        # Old prediction was wrong, new prediction is wrong
        elif top_old_prediction != true_prediction and top_generated_prediction != true_prediction:
            num_incorrect_to_incorrect += 1
        # Old prediction was right, new prediction is right
        elif top_old_prediction == true_prediction and top_generated_prediction == true_prediction:
            num_correct_to_correct += 1

    print("Num Ignored Classifications " + str(len(old_score_files)))

    # Now iterate through the remaining old files, and see how many opportunities we missed with reclassification.
    for old_score_file in old_score_files:
        # Get the correct activity
        true_prediction = int(old_score_file[ old_score_file.index("act_") + 4 : old_score_file.index("_index") ])
        old_score_filepath = old_score_dir + "/" + old_score_file
        old_scores = loadArrayFromFile(old_score_filepath)
        top_old_prediction = np.argmax(old_scores)

        # If the predictions match
        if top_old_prediction == true_prediction:
            num_correct += 1
        # if the predictions don't match
        else:
            num_incorrect += 1

    print("Num Correct without Reclassification: " + str(num_correct))
    print("Num Incorrect without Reclassification: " + str(num_incorrect))

    # Remember, this may be more than the actual number of classifications
    #  because each old classification may have multiple possible modality combinations
    #  added to it for reclassification.
    print("\n\nNum Reclassifications: " + str(len(new_score_files)))
    print("\tIncorrect to Correct: " + str(num_incorrect_to_correct))
    print("\tCorrect to Incorrect: " + str(num_correct_to_incorrect))
    print("\tIncorrect to Incorrect: " + str(num_incorrect_to_incorrect))
    print("\tCorrect to Correct: " + str(num_correct_to_correct))

    # print("\n\n Metrics For Correct To Incorrect: ")
    # pprint(correct_to_incorrect_dicts)
    #
    print("\n\n Metrics For Incorrect To Correct: ")
    pprint(incorrect_to_correct_dicts)

    return [num_incorrect_to_correct, num_correct_to_incorrect, num_incorrect_to_incorrect, num_correct_to_correct, \
    len(new_score_files), num_incorrect, len(old_score_files), incorrect_to_correct_dicts]

def do_reclassification(reclassified_scores_dir, statistics_dir, total_epochs, num_activities, lam=0.25):

    subprocess.call('mkdir ' + reclassified_scores_dir, shell=True)

    print("Loading Dictionary Values for confidence and penalities")

    # Also load the dictionaries containing the confidences and penalties:
    # This is a dictionary containing keys of id XYYY, where X is the activity label and YYY is the sensor modality combination,
    #   and values are a list of two items - summed confidences and number of datapoints.  You can divide these to get average confidence [0-1]
    confidences_dict = loadDictionaryFromFile(statistics_dir + "/model_confidences_augmented_validation.pkl")

    #pprint(confidences_dict)
    # This is a dictionary containing keys of modality_dirs, and the items are the dictionaries of activity : penalty,
    #  Each penalty is in the range [0-1]
    penalties_dict = loadDictionaryFromFile(statistics_dir + "/penalties_validation.pkl")
    #pprint(penalties_dict)

    # Also load the feature generators, and return them here in a list.
    #  Also load the corresponding feature folders that we will use.
    modality_dirs = ["image_chunks", "imu_chunks", "audio_chunks"]  #MAKE SURE THIS MATCHES WITH THE TRAINING SECTION and EVALUATION
    model_list = loadFeatureGenerators(modality_dirs, total_epochs)
    feature_folder_dirs = findFeatureFolderListFromModalities(modality_dirs)

    # Load the feature fusion model
    fusion_model = load_model("saved_model/fusion/fusion_model/" + "epoch_000" + str(total_epochs) + ".h5")

    # Iterate through each combination feature and find the best modelContribution.
    combo_files = os.listdir("features/augmented_testing")
    old_score_folder = "scores/augmented_testing/"  #Folder containing the scores of previous classification.

    print("Producing Reclassification Scores...")

    possible_reclassification_count = 0

    for filename in combo_files:

        #print(filename)

        # This best_modality_combo is of the form [0,1,0] where 1 indicates a modality is present (i.e. modality 2 is present)
        # Figure out what the best model combination for regenerating features.
        modelContributors, num_reclassification_options = FindBestModelContribution(filename, modality_dirs, confidences_dict, \
        penalties_dict, lam=lam, num_activities=num_activities)

        #print(modelContributors)

        #break
        for modalities_with_new_features, modalities_to_regenerate, in modelContributors:
            # It has been decided that the original classification is most reliable.
            if not modalities_with_new_features:
                continue

            # We are redoing a classification, so produce the input data we need.
            input_data, renamed_file = produce_concatenated_features(old_score_folder, filename, model_list, feature_folder_dirs, modalities_with_new_features, \
            modalities_to_regenerate)

            # Use the input data to get output from the model, and save it to file.
            produce_scores(input_data, fusion_model, reclassified_scores_dir, renamed_file)

        # We add how many possible reclassifications we can have for this file combo (remember, we can only add modalities)
        possible_reclassification_count += num_reclassification_options

    print("Total Possible Reclassifications: " + str(possible_reclassification_count))
    return possible_reclassification_count

def produce_scores(input_data, fusion_model, reclassified_scores_dir, renamed_file):

    #print(input_data.shape)
    #print(renamed_file)
    output = fusion_model.predict(input_data, batch_size=1)
    #print(output)
    #print(output.shape)
    #print("Saving New Score to " + str(renamed_file))
    saveArrayToFile(output, reclassified_scores_dir + "/" + renamed_file)

#  Use the ModelContribution calculation to determine if you should fuse/how you should fuse
#  If you ought to fuse, also record the following:
#      - what the highest scoring ModelContribution is
#      - using the recommended sensor combinations, run classification: what are the new scores?
#      - what is the new activity classification (prediction)?
#  Basically, produce the input needed for runnign classification.
#  It also returns the new name of the score file.
def produce_concatenated_features(old_score_folder, filename, model_list, feature_folder_dirs, modalities_with_new_features, \
modalities_to_regenerate):

    new_input_data = []

    # Convert the shared feature filename into the individual feature file naming format.
    #i.e. from act_0_index_2_combo_010.features.npy into act_0_index_2.features.npy
    feature_filename = filename[: filename.index("_combo") ] + ".features.npy"

    # Convert the shared feature filename into the score naming format
    #i.e. from act_0_index_2_combo_010.features.npy into act_0_index_2_combo_010.scores.npy
    score_filename = filename.replace("features", "scores")

    # Convert the score name into the new score file name
    #i.e. from act_0_index_2_combo_010.scores.npy into act_0_index_2_combo_010_newcombo_111.scores.npy
    new_score_filename = score_filename[: score_filename.index(".scores")]
    # This produces the new combination of modalities that we are fusing together (both generated and new)
    new_combination = [modalities_with_new_features[i] or modalities_to_regenerate[i] for i in range(0, len(modalities_with_new_features))]
    new_score_filename += "_newcombo_" + "".join(str(i) for i in new_combination) + ".scores.npy"

    # Iterate through each modality.
    for i in range(0, len(model_list)):

        # We should take the input for this new modality.
        if modalities_with_new_features[i]:
            input_data = loadArrayFromFile(feature_folder_dirs[i] + "/" + feature_filename)
            new_input_data.append(input_data)

        # We should regenerate the feature for this modality.
        elif modalities_to_regenerate[i]:
            score_data = loadArrayFromFile(old_score_folder + "/" + score_filename)
            #print(score_data)
            #print(np.array(modalities_to_regenerate).shape)
            score_data = np.hstack([score_data, np.expand_dims(np.array(modalities_to_regenerate), axis=0)])

            #print(score_data.shape)
            #score_data = score_data.reshape(1, score_data.shape[0])

            input_data = model_list[i].predict(score_data, batch_size=1)
            new_input_data.append(input_data)

        # We don't have new input, and we don't have features to regenerate.
        #  This means it's just a zero array of feature size for the modality.
        else:
            input_data = np.zeros_like(loadArrayFromFile(feature_folder_dirs[i] + "/" + feature_filename))
            new_input_data.append(input_data)

    new_input_data = np.hstack(new_input_data)
    return new_input_data, new_score_filename



# Basically, takes a list of modalities and outputs the matching folder dirs for them.
def findFeatureFolderListFromModalities(modality_dirs):

    feature_folder_list = []
    for modality in modality_dirs:
        feature_folder_list.append("features/testing/" + modality)

    return feature_folder_list


# Returns each feature generator in a list.  Make sure the order matches with how we check our features.
def loadFeatureGenerators(modality_dirs, total_epochs=10):

    model_list = []
    for modality in modality_dirs:

        model = load_model("saved_model/generator/" + modality + "_generator/" + "epoch_000" + str(total_epochs) + ".h5")
        model_list.append(model)

    return model_list


# Compare the model contribution of a new modality - also checks if this modality
#  Classified activity is the integer label produced by a classification.
#  new_modality_list is a list of modalities that we can add to this combo. (should be of the form [0,1,0] meaning modality 2 is present.)
#  old_modality_list is a list of modalities that was used for this classification.
#  Assumes that the new modality list is a valid combination to add to the old modality list
# lam is the lambda hyperparameter for changing the effect of the penalty
def calculateModelContribution( classified_activity, modality_dirs, new_modality_list, old_modality_list ,confidences_dict, penalties_dict, lam, num_activities=6):

    combined_modality_list = [new_modality_list[i] or old_modality_list[i] for i in range(0, len(old_modality_list))]

    modelContribution = 0

    for activity in range(0, num_activities):

        if classified_activity != activity:  # We only add to the model contribution if we could get a different decision out of it.

            new_id = str(activity) + "".join(str(i) for i in new_modality_list)
            old_id = str(classified_activity) + "".join(str(i) for i in old_modality_list)

            penalty = 0
            # old modality list indicates what modalities we would have to regenerate.
            for m_index in range(0, len(old_modality_list)):
                modality_present = old_modality_list[m_index]  # Will be 1 if we should generate features, 0 to ignore.
                penalty += lam * modality_present * penalties_dict[modality_dirs[m_index]][activity]

            # print(confidences_dict[new_id])
            # print(confidences_dict[old_id])
            # print(penalty)

            # Each confidence entry is a pair of summed confidences and number of datapoints.
            new_confidence = confidences_dict[new_id][0]/confidences_dict[new_id][1]
            old_confidence = confidences_dict[old_id][0]/confidences_dict[old_id][1]

            modelContribution += new_confidence - old_confidence - penalty

    return modelContribution


# Produce a list of possble present modalities into the old modality set.
#  For example, if we have an old list [0,1,0], we can get new modality combinations:
#   [1,1,0], [0,1,1], and [1,1,1]
def producePossibleCombos(old_modality_list):
    #print(old_modality_list)

    # This produces the list of all possible combinations of modalities.
    # To save computation, we can call this earlier on and just pass the entire list for pruning later.
    #  We essentially have to prune from this list whichever ones dont have at least the old modalities present.
    allCombos = createAugmentedDataCombinations(old_modality_list)

    possible_combos = []

    for combo in allCombos:

        int_combo = [int(x) for x in combo]  #Converts true/false into 1 and 0
        isInvalid = False

        #Iterate through each combination, and continue if any of the old modalities aren't present.
        #  Basically, an old modality will have value 1 - if it is not present in the new modality,
        #   then the new modality has value 0.
        for i in range(0, len(combo)):
            if old_modality_list[i] > int_combo[i]:
                isInvalid = True
                break

        # If this combination is valid, then we add it to the possible_combos.
        if not isInvalid:
            possible_combos.append(int_combo)

    # Also remove the matching combination from the possible_combos
    possible_combos.remove(old_modality_list)
    #print(possible_combos)
    return possible_combos



# The way this works is we go through all the combinations of features that already exist,
#  and we find what modalities are missing for each one.  We calculate the best new modality combination
#   for that feature combo.
# Given the filename and path for an augmented file, find what combinations we can calculate the model contribution for.
#  filename is of the format act_0_index_5_combo_010.features.npy
#  The lambda is the hyperparameter for tweaking the penality value.

# However, we are not just returning the best model combination
#  We also need to find which modalities are new, and which ones are old.
# This actually returns a list of the modalities that we can recombine.
#  If topbest = True, we are assuming that we have all modalities, and can just take the best.
def FindBestModelContribution(filename, modality_dirs, confidences_dict, penalties_dict, lam=0.25, num_activities=6, topBest=False):

    old_modality_list = filename[filename.index("combo_") + 6 : filename.index(".features")]
    old_modality_list = [int(x) for x in old_modality_list]  # Convert the string "000" into [0,0,0]
    classified_activity = int(filename[filename.index("act_") + 4 : filename.index("_index")])

    new_modality_combos = producePossibleCombos(old_modality_list)

    # This tells us how many reclassifications we can choose to make
    total_possible_reclassifications = len(new_modality_combos)

    best_model_contribution = 0
    best_model_combo = []
    best_added_modality_list = []

    contributing_models = []

    for new_modality_list in new_modality_combos:
        modelContribution = calculateModelContribution(classified_activity, modality_dirs, new_modality_list, old_modality_list , \
        confidences_dict, penalties_dict, lam=lam, num_activities=num_activities)

        if modelContribution > 0: # new modality can actually contribute, so add the relevant data to the list.
            # This is the modality we are adding - therefore, we need to take the features from this modality
            added_modality_list = [new_modality_list[i] - old_modality_list[i] for i in range(0, len(new_modality_list))]
            contributing_models.append( (added_modality_list, old_modality_list)  )

        if topBest and modelContribution > best_model_contribution:
            best_model_combo = new_modality_list
            best_model_contribution = modelContribution
            best_added_modality_list = added_modality_list

    # We couldn't find a better contribution.
    if topBest and best_model_contribution == 0:
        return [],[]

    if topBest:
        # This is the modality we had before, and therefore will have to regenerate.
        return new_modality_list, old_modality_list
    else:
        return contributing_models, total_possible_reclassifications

def main():
    # If you want to redo the entire train/prediction/evaluation pipeline,
    #  delete the features, logs, saved_model, scores, and statistics folders.
    train()

    max_lambda = 3.0
    lambdas = np.arange(0.0, max_lambda, 0.1)
    #lambdas = [1.2, 2.2]

    i2c = []  # % of reclassifications that change from incorrect to correct
    c2i = [] #  % of classifications that change from correct to incorrect
    i2i = [] # % of classifications that change from incorrect to incorrect
    c2c = [] #  % of classifications that change from correct to correct
    tr_tpr = []  # % of total reclassifications over total possible reclassifications
    icc_ic = [] # % of incorrect classifications over ignored clssificatinos
    #tpr = [] # number of total possible reclassifications.

    modality_names = ["audio", "image", "imu"]
    added_modality_i2c = [[] for modality in modality_names]  #  % of incorrect to correct classifications that are each sensor modality (contains a list per modality)


    for lam in lambdas:

        print("Evaluating on Lambda=" + str(lam))

        if os.path.exists("scores/reclassified_testing"):
            # Delete the old classification folder, redo classification with new lambda
            subprocess.call("rm -rf scores/reclassified_testing", shell=True)

        total_possible_reclassifications = prediction(lam)
        #tpr.append(total_possible_reclassifications)

        overall_counts = evaluate()  # This is an array of num i2c, c2i, i2i, c2c, and reclassifications
        i2c.append(overall_counts[0]/overall_counts[4] *100)
        c2i.append(overall_counts[1]/overall_counts[4]*100)
        i2i.append(overall_counts[2]/overall_counts[4]*100)
        c2c.append(overall_counts[3]/overall_counts[4]*100)
        tr_tpr.append(overall_counts[4]/total_possible_reclassifications*100)
        icc_ic.append(overall_counts[5]/overall_counts[6]*100)

        # This goes through the list of modalities and corresponding instances of incorrect to correct per activity,
        #   sums the total number of incorrect to correct for this modality, and calculates the % of i2c for this modality / total i2c

        incorrect_to_correct_added_modalities = overall_counts[7][2]
        for i in range(0, len(modality_names)):

            modality = modality_names[i]

            if modality in incorrect_to_correct_added_modalities:
                summed_instances = sum(incorrect_to_correct_added_modalities[modality])  # Number of correctly reclassified instances for this modality
                percentage = summed_instances/overall_counts[0]*100
                added_modality_i2c[i].append(percentage)
            else:
                added_modality_i2c[i].append(0)


    #print(added_modality_i2c)

    plt.close()

    plt.title("Effect of Lambda on Reclassification Scores")
    plt.plot(lambdas, i2c, "b--", lambdas, c2i, "r--", lambdas, i2i, "y--", lambdas, c2c, "g--")
    plt.ylabel("% of reclassifications")
    plt.xlabel("values of lambda")
    plt.legend(["% Incorrect to Correct", "% Correct to Incorrect", "% Incorrect to Incorrect", "% Correct to Correct"])
    plt.axis([0.0, max_lambda, 0, 100])
    plt.show()

    plt.title("Effect of Lambda on Ratio of Reclassification")
    plt.plot(lambdas, tr_tpr)
    plt.ylabel("% Actual Reclassifications over Total Possible Reclassifications")
    plt.xlabel("values of lambda")
    plt.axis([0.0, max_lambda, 0, 100])
    plt.show()

    plt.title("Effect of Lambda on Percentage of Missed Opportunities")
    plt.plot(lambdas, icc_ic)
    plt.ylabel("% Incorrect Ignored Classifications over Total Ignored Classifications")
    plt.xlabel("values of lambda")
    plt.axis([0.0, max_lambda, 0, 100])
    plt.show()

    # This is composed of 3 lists, one for each modality and each list being a % for each lambda.
    plt.title("Effect of Lambda on on Incorrect to Correct Classification Contributed by Modality")
    plt.plot(lambdas, added_modality_i2c[0], "b", lambdas, added_modality_i2c[1], "r", lambdas, added_modality_i2c[2], "g")
    plt.ylabel("% Incorrect to Correct for each Modality")
    plt.xlabel("values of lambda")
    plt.legend(modality_names)
    plt.axis([0.0, max_lambda, 0, 100])
    plt.show()



    plt.close()

main()
