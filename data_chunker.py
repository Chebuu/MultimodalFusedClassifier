from numpy import random
import numpy as np

import os
import pandas as pd
import subprocess
import numpy as np
from math import floor

from data_utils import loadImageChunk,loadIMUChunk, loadAudioChunk, saveArrayToFile, loadArrayFromFile, get_num_chunks


# def concatenateChunk(sequence=1, chunk_index=0, chunk_size=1, zero_out={"image": False, "imu": False, "audio": False}):
#
#     # act_str = "act" + str(act).rjust(2, "0")
#     seq_str = "seq_" + str(sequence)
#     # seq_str_imu = "seq" + str(sequence).rjust(2, "0")
#
#     imu_file_endings = ["_id_la", "_id_lg", "_id_ra", "_id_rg"]
#     imu_chunks = []
#
#     audio_filename = seq_str + "_channel_0.wav"
#
#     #try:
#     image_chunk = loadImageChunk("final_dataset/images_final/" + seq_str, video_id=sequence, chunk_index=chunk_index, chunk_size=chunk_size)
#     #imu_chunk = loadIMUChunk("sensor/""act01seq02"".csv", samplingfreq=10, chunk_index=0)
#
#     # Stack IMU chunks together and save
#     for file_ending in imu_file_endings:
#         imu_filename = seq_str + file_ending
#         imu_chunks.append(loadIMUChunk("final_dataset/imu_final/" + imu_filename, samplingfreq=25, chunk_index=chunk_index, chunk_size=chunk_size))
#     imu_chunk = np.hstack(imu_chunks)
#
#     audio_chunk, audio_features_chunk = loadAudioChunk("final_dataset/audio_final/" + audio_filename, bitrate=16000, chunk_index=chunk_index, chunk_size=chunk_size)
#
#     print(image_chunk.shape)
#     print(imu_chunk.shape)
#     print(audio_chunk.shape)
#     print(audio_features_chunk.shape)
#
#     if zero_out["image"]:  #We want to zero out the image chunk
#         image_chunk = np.zeros_like(image_chunk)
#     if zero_out["imu"]:  #We want to zero out the imu chunk
#         imu_chunk = np.zeros_like(imu_chunk)
#     if zero_out["audio"]:  #We want to zero out the audio chunk
#         audio_chunk = np.zeros_like(audio_chunk)
#
#     combined_chunk = np.array(image_chunk)
#
#     # IMU data is #samples x #channels.  We should split this to match the number of samples for images
#     num_image_samples = image_chunk.shape[0]
#
#     num_imu_samples = imu_chunk.shape[0]
#
#     # Basically, reshape the chunk of IMU data to be combined with the image chunk
#     if num_imu_samples < num_image_samples:
#         imu_samples_per_image = num_image_samples//num_imu_samples  # Use this to determine how many copies we need
#         # POTENTIAL PROBLEM: If these do not divide evenly, then we will end up with more IMU samples than image samples
#
#         # Repeat this along an axis (i.e. [1,2,3] becomes [1, 1, 2, 2, 3, 3]) and reshape to be the same number of dims as combined_chunk
#         current_imu_chunk = np.repeat(imu_chunk, imu_samples_per_image, axis=0).reshape((num_image_samples, imu_chunk.shape[1], 1, 1))
#         # Broadcast this shape to be the same as the combined_chunk except along the axis we are appending to.
#         current_imu_chunk = np.broadcast_to(current_imu_chunk, (current_imu_chunk.shape[0], current_imu_chunk.shape[1], combined_chunk.shape[2], combined_chunk.shape[3]))
#         combined_chunk = np.append( combined_chunk, current_imu_chunk, axis=1 )
#
#     else:
#         imu_samples_per_image = num_image_samples//num_imu_samples
#
#         # Still need to come up with a solution for this.
#
#
#     print("combibned chunke")
#     print(combined_chunk.shape)
#     # Audio samples is #samples.  Split this to match the number of samples for images
#     num_audio_samples = audio_chunk.shape[0]
#
#     audio_samples_per_image = num_audio_samples//num_image_samples
#     # We are trying to match segments of audio with an image.
#     # If the number of samples is not divisible with the number of images, we just cut off/remove whatever is left
#     current_audio_chunk = audio_chunk[:-(num_audio_samples%num_image_samples),].reshape((num_image_samples, num_audio_samples//num_image_samples, 1, 1))
#     current_audio_chunk = np.broadcast_to(current_audio_chunk, (current_audio_chunk.shape[0], current_audio_chunk.shape[1], combined_chunk.shape[2], combined_chunk.shape[3]))
#     combined_chunk = np.append(combined_chunk, current_audio_chunk, axis=1)
#
#     return combined_chunk  #Returns the combined chunk for all modalities
#
#     #except:
#         #print("*****BLWBEFIWEBFBWIEBF*****")

def concatenateChunk(chunk_filename, audio_dir = "audio_image_chunks", video_dir = "image_chunks", imu_dir = "imu_chunks", save_dir="concatenated_chunks"):

    image_chunk, _ = loadChunkAndLabelFromFile(video_dir + "/" + chunk_filename)
    audio_chunk, _ = loadChunkAndLabelFromFile(audio_dir + "/" + chunk_filename)
    imu_chunk, _ = loadChunkAndLabelFromFile(imu_dir + "/" + chunk_filename)

    #print(image_chunk.shape)

    audio_chunk = audio_chunk.reshape(audio_chunk.shape[0], 1, audio_chunk.shape[1], audio_chunk.shape[2], audio_chunk.shape[3])
    #imu_chunk = imu_chunk.reshape(imu_chunk.shape[0], imu_chunk.shape[1], imu_chunk.shape[2], 1, 1)

    # This maps the imu chunk to 1,25,12,5
    imu_chunk = np.expand_dims(imu_chunk, axis=4)
    imu_chunk = np.broadcast_to(imu_chunk, (imu_chunk.shape[0], imu_chunk.shape[1], imu_chunk.shape[2], 5))
    # This maps the imu chunk to 1,25,60, 1, 1
    imu_chunk = imu_chunk.reshape((imu_chunk.shape[0], imu_chunk.shape[1], imu_chunk.shape[2]*imu_chunk.shape[3], 1, 1))
    # This pushes the imu chunk to be the same shape as the audio, except for the 25 samples
    imu_chunk = np.broadcast_to(imu_chunk, (imu_chunk.shape[0],imu_chunk.shape[1],imu_chunk.shape[2], audio_chunk.shape[3], audio_chunk.shape[4]))

    # This combines the audio and imu chunks.
    combined_chunk = np.append(imu_chunk, audio_chunk, axis=1)
    combined_chunk = np.reshape(combined_chunk,(combined_chunk.shape[0],combined_chunk.shape[1], combined_chunk.shape[4], combined_chunk.shape[2], \
    combined_chunk.shape[3]))

    #print(combined_chunk.shape)

    new_combined_chunk = []

    # For each layer in the combined chunk, reshape it to match with the image chunks
    for i in np.arange(combined_chunk.shape[1]):

        padded_chunk = np.zeros((1,1,3,90,160))
        for j in np.arange(combined_chunk.shape[2]):

            #print(padded_chunk.shape)
            padded_chunk[0,0,j] = np.pad(combined_chunk[0][i][j], ((15,15), (64, 64)), 'reflect')

        new_combined_chunk.append(padded_chunk)

    new_combined_chunk = np.vstack( new_combined_chunk)

    # Reshape to match the image chunk
    new_combined_chunk = np.reshape(new_combined_chunk, (1,26,90,160,3))

    # Concatenate new chunk with video/image chunk
    new_combined_chunk = np.append( image_chunk, new_combined_chunk, axis=1 )

    saveArrayToFile(new_combined_chunk, save_dir + "/" + chunk_filename)

# Assumes all invdividual modality chunks have been created.
def createAllConcatenatedChunks():

    chunk_dir = "concatenated_chunks"
    if not os.path.exists(chunk_dir):
        subprocess.call('mkdir ' + chunk_dir, shell=True)

        # Find all the chunks that we need to concatenate
        #  Remember that all of them are named the same way, so we just use any folder.
        chunks_to_concatenate = os.listdir("audio_image_chunks")

        for chunkfilepath in chunks_to_concatenate:
            print("Concatenating Chunk for " + str(chunkfilepath))
            concatenateChunk(chunkfilepath, save_dir = chunk_dir)

def generate_chunks(parent_dir, files, batch_size=64):

    isAutoencoder = False
    if parent_dir == "concatenated_chunks":
        isAutoencoder = True

    while True:

        batch_files = random.choice(a= files, size=batch_size)
        batch_data, batch_labels = loadChunksFromFileList(parent_dir = parent_dir, filelist=batch_files)

        if parent_dir == "audio_chunks":
            batch_data = np.expand_dims(batch_data, axis=2)

        if parent_dir == "imu_chunks":
            _,x_1,x_2, = batch_data.shape
            batch_data = batch_data.reshape((batch_size, x_1 * x_2, 1))
            #print(batch_data.shape)

        if parent_dir == "concatenated_chunks":

            batch_data = np.squeeze(batch_data)
            batch_data = batch_data[:,:-1,:,:,:]
            #print(batch_data.shape)

        #print(batch_data.shape)
        #print(batch_labels.shape)

        if isAutoencoder:
            yield(batch_data, batch_data)
        else:
            yield( batch_data, batch_labels )
#current_chunk = concatenateChunk(act=1, sequence=1, chunk_index=0, chunk_size=1)

from itertools import permutations

# This creates the data augmentation combinations by selecting what modality to zero out.
def createAugmentedDataCombinations(modalities = ["image", "imu", "audio"]):

    combo_list = []  #This will contain dictionaries of {"image": True, ...}

    starting_permutation = [False for x in modalities]

    all_combos = []
    #all_combos.append([False for x in modalities])  #Some issue with cloning causes this
    # original data to be changed, hence we dont use starting_permutation

    # We add one true to the list and generate permutations for it
    for i in range(0, len(starting_permutation)-1):
        starting_permutation[i] = True
        permutation_list = set(permutations(starting_permutation))
        permutation_list = [ list(x) for x in permutation_list ]

        all_combos.extend( permutation_list ) #Set is used to keep only unique elements

    all_combos.append( [True for x in modalities] )

    # We create the dictionaries we need [ { modality: boolean, ...}, { modality: boolean...} ...]
    # for config in all_combos:
    #     current_zero_combo = { modalities[i]: config[i] for i in range(0, len(modalities)) }
    #     combo_list.append(current_zero_combo)

    return all_combos

def saveChunkToFile(parent_dir, datachunk, sequence, chunk_index, chunk_size):

    if not os.path.exists(parent_dir):
        subprocess.call('mkdir ' + parent_dir, shell=True)

    filename = "seq_" + str(sequence) + "_ci_" + str(chunk_index) + "_cs_" + str(chunk_size) + ".npy"
    print("Saving Datachunk: " + filename)
    saveArrayToFile(datachunk, parent_dir + "/" + filename)


# This just maps a sequence to a label.
def findLabelFromSequence(sequence):
    # Sequence 0, 11 are walking 0
    # Sequence 1, 6, 12 are standing 1
    # Sequence 2, 7, 13 are sitting 2
    # Sequence 3, 8, 14 are writing 3
    # Sequence 4, 9, 15 are playing on phone 4
    # Sequence 5, 10, 16 are reading 5

    label_sequence_mapping = {0:0, 11:0, 1:1, 6:1, 12:1, 2:2, 7:2, 13:2, 3:3, 8:3, \
    14:3, 4:4, 9:4, 15:4, 5:5, 10:5, 16:5}

    return label_sequence_mapping[sequence]



def loadChunkAndLabelFromID(parent_dir, sequence, chunk_index, chunk_size):

    filename = "seq_" + str(sequence) + "_ci_" + str(chunk_index) + "_cs_" + str(chunk_size) + ".npy"
    label = findLabelFromSequence(sequence)
    return loadArrayFromFile(parent_dir + "/" + filename), label

def loadChunkAndLabelFromFile(filepath):

    sequence = int(filepath[filepath.index("seq_") + 4: filepath.index("_ci")])

    label = findLabelFromSequence(sequence)
    return loadArrayFromFile(filepath), label


# Produces a list of filenames used for training and testing sets
# Keep in mind that the filename does not include the parent directory
#  the split ratio determines what proportion of the total data should be training/validation
#  omit_sequences is a list of the sequences we want to omit from the training/validation list
#  omit_labels is a list of the labels we want to omit from the training/validation list
def loadTrainingTestingFiles(parent_dir, split_ratio=[0.8,0.2,0.0], omit_sequences=[], omit_labels=[]):

    print("Organizing Files for Training/Validation/Testing")

    chunk_filenames = os.listdir(parent_dir)

    useable_files = []
    training_files = []
    validation_files = []
    testing_files = []

    # Initialize a dictionary of labels that have count of zero.
    #  We will use this to determine if we have reached the appropriate proportion of data.
    label_count = {}  # This is how many chunks for each label we want for each set (training, validation, testing)
    label_total = {}  # This is how many chunks we have total per label
    for i in range(0, 6):
        if i not in omit_labels:
            label_count[i] = [0,0,0]
            label_total[i] = 0

    # We find the list of useable chunks and append them to the list.
    for chunk_filename in chunk_filenames:

        sequence = int(chunk_filename[chunk_filename.index("seq_") + 4 : chunk_filename.index("_ci")])
        label = findLabelFromSequence(sequence)

        # We are omitting this from consideration, so just keep going.
        if sequence in omit_sequences or label in omit_labels:
            continue
        else:
            useable_files.append(chunk_filename)
            label_total[label] += 1

    #print(label_total)

    # Find the amount of data we need for each label for each training and validation partition.
    num_datapoints = len(useable_files)
    num_labels = len(label_count.keys())

    # Get the number of datapoints we should have for each label and set
    for label, total in label_total.items():

        num_training_datapoints = total * split_ratio[0]
        num_validation_datapoints = total * split_ratio[1]
        num_testing_datapoints = min(total - num_training_datapoints - num_validation_datapoints, total * split_ratio[2])



        # max_label_training_datapoints = num_training_datapoints // num_labels
        # max_label_validation_datapoints = num_validation_datapoints // num_labels
        # max_label_testing_datapoints = num_testing_datapoints // num_labels
        label_count[label][0] = floor(num_training_datapoints)
        label_count[label][1] = floor(num_validation_datapoints)
        label_count[label][2] = floor(max(0, num_testing_datapoints))

    print("Label Count [training, validation, testing]: " + str(label_count))

    # We use the previously found useable chunk files and create training/validation/testing set.
    for chunk_filename in useable_files:

        sequence = int(chunk_filename[chunk_filename.index("seq_") + 4 : chunk_filename.index("_ci")])
        label = findLabelFromSequence(sequence)

        if label_count[label][0] > 0:
            label_count[label][0] -= 1
            training_files.append(chunk_filename)
        elif label_count[label][1] > 0:
            label_count[label][1] -= 1
            validation_files.append(chunk_filename)
        elif label_count[label][2] > 0:
            label_count[label][2] -= 1
            testing_files.append(chunk_filename)

    return training_files, validation_files, testing_files

#  Given a list of .npy files, load them into np arrays and return the data.
def loadChunksFromFileList(parent_dir, filelist):
    chunk_data = []
    chunk_labels = []

    for filename in filelist:
        chunk, label = loadChunkAndLabelFromFile(parent_dir + "/" + filename)
        chunk_data.append(chunk)
        chunk_labels.append(label)

    if chunk_data:  # We actually have data, so vstack the lists.
        chunk_data = np.vstack( chunk_data )
        chunk_labels = np.vstack( chunk_labels )
        #print("Chunk Data Shape: " + str(chunk_data.shape))
        #print("Chunk Labels Shape: " + str(chunk_labels.shape))

    return chunk_data, chunk_labels


# Actually loads the numpy data from file, returns very large data for training, validation, and testing.
def loadTrainingTestingSets(parent_dir, split_ratio=[0.8,0.2,0.0], omit_sequences=[], omit_labels=[]):

    training_files, validation_files, testing_files = \
    loadTrainingTestingFiles(parent_dir, split_ratio, omit_sequences, omit_labels)

    training_data, training_labels = [], []
    validation_data, validation_labels = [], []
    testing_data, testing_labels = [], []

    print("Loading Chunks for Training")

    for filename in training_files:

        chunk, label = loadChunkAndLabelFromFile(parent_dir + "/" + filename)
        training_data.append(chunk)
        training_labels.append(label)

    print("Loading Chunks for Validation")
    for filename in validation_files:

        chunk, label = loadChunkAndLabelFromFile(parent_dir + "/" + filename)
        validation_data.append(chunk)
        validation_labels.append(label)

    print("Loading Chunks for Testing")
    for filename in testing_files:

        chunk, label = loadChunkAndLabelFromFile(parent_dir + "/" + filename)
        testing_data.append(chunk)
        testing_labels.append(label)

    if training_data:
        training_data = np.vstack( training_data )
        training_labels = np.vstack( training_labels )
        print("Training Data Shape: " + str(training_data.shape))
        print("Training Labels Shape: " + str(training_labels.shape))
    if validation_data:
        validation_data = np.vstack( validation_data )
        validation_labels = np.vstack( validation_labels )
        print("Validation Data Shape: " + str(validation_data.shape))
        print("Validation Labels Shape: " + str(validation_labels.shape))
    if testing_data:
        testing_data = np.vstack( testing_data )
        testing_labels = np.vstack( testing_labels )
        print("Testing Data Shape: " + str(testing_data.shape))
        print("Testing Labels Shape: " + str(testing_labels.shape))

    return training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels


#  CUSTOMIZE THIS FOR THE DATA:
#    We have to change how we iterate through the files
#    Also save the chunked data as np array files, and retrieve it later.
#      That way we don't have to create them every time.
def save_all_chunks(augmented_data = False):


    data_combinations = [{"image": False, "imu": False, "audio": False}]
    if augmented_data:
        data_combinations = createAugmentedDataCombinations()

    # How many sequences do we have?
    for i in range(0,17):

        # Find how many chunks we can iterate through
        num_chunks = get_num_chunks("final_dataset/images_final/seq_" + str(i))

        # Iterate through each chunk and concatenate
        for j in range(0, num_chunks):

            print("Producing Chunks for Sequence " + str(i) + " and Chunk ID " + str(j))

            try:  # If we bump into an assertion error, that means we are out of data, so move on.
                #current_concatenated_chunk = concatenateChunk(sequence=i, chunk_index=j, chunk_size=1)
                #current_concatenated_chunk = np.expand_dims(current_concatenated_chunk, axis=0)


                # Also save Chunks for Images, IMU, and Audio
                seq_str = "seq_" + str(i)
                imu_file_endings = ["_id_la", "_id_lg", "_id_ra", "_id_rg"]
                imu_chunks = []

                audio_filename = seq_str + "_channel_0.wav"

                image_chunk = loadImageChunk("final_dataset/images_final/" + seq_str, video_id=i, chunk_index=j, chunk_size=1)
                #imu_chunk = loadIMUChunk("sensor/""act01seq02"".csv", samplingfreq=10, chunk_index=0)

                for file_ending in imu_file_endings:
                    imu_filename = seq_str + file_ending
                    imu_chunks.append(loadIMUChunk("final_dataset/imu_final/" + imu_filename, samplingfreq=25, chunk_index=j, chunk_size=1))
                imu_chunk = np.hstack(imu_chunks)

                audio_chunk, audio_image_chunk = loadAudioChunk("final_dataset/audio_final/" + audio_filename, bitrate=16000, chunk_index=j, chunk_size=1)


            #current_chunk = current_chunk[:,1:-1,:,:,]  #Remove one pixel from the image
            #print("Chunk Shape: " + str(current_chunk.shape))

                #saveChunkToFile("concatenated_chunks", current_concatenated_chunk, sequence=i, chunk_index=j, chunk_size=1)
                saveChunkToFile("image_chunks", image_chunk, sequence=i, chunk_index=j, chunk_size=1)
                saveChunkToFile("imu_chunks", imu_chunk, sequence=i, chunk_index=j, chunk_size=1)
                saveChunkToFile("audio_chunks", audio_chunk, sequence=i, chunk_index=j, chunk_size=1)
                saveChunkToFile("audio_image_chunks", audio_image_chunk, sequence=i, chunk_index=j, chunk_size=1)

                #break
            except Exception as error:
                if str(error):
                    print("Data Chunker Error: " + str(error))
                continue

        #break
#save_all_chunks()
