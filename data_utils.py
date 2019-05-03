
import os
import numpy as np
import librosa
import cv2
import pickle

#TODO: Normalization??

# Get Image data according to specifications
# Here we can also resize the image
def loadImage(filepath):
    img = cv2.imread(filepath)
    return img

# Get all images for each video
# The chunk size determines how many images we choose to have
# The chunk index determines which chunk we start from.
# i.e. a chunk index of 1 and a chunk size of 1 and fps of 30 means we take 1*30 jpgs starting
#  from the 31st image in the folder.
def loadImageChunk(folder_dir, video_id, chunk_index = 0, chunk_size=1, fps=15):

    file_dir = folder_dir

    image_files = os.listdir(file_dir)
    image_files.sort()
    num_files = len(image_files)

    file_id = chunk_index*chunk_size*fps
    assert(file_id + chunk_size*fps <= num_files)  #Check if we still have enough data

    file_chunk_paths = image_files[file_id : file_id + chunk_size*fps]


    image_chunks = loadImage(file_dir + "/" + file_chunk_paths[0])
    image_chunks = np.expand_dims(image_chunks, axis=0)

    for i in range(1, chunk_size*fps):
        image_to_append = loadImage(file_dir + "/" + file_chunk_paths[i])
        image_to_append = np.expand_dims(image_to_append, axis=0)
        image_chunks = np.concatenate( (image_chunks, image_to_append) )

    #print(image_chunks.shape)
    return image_chunks

# Get the number of chunks for this folder - how many files in each chunk?
# Used specifically for the images directory
def get_num_chunks(folder_dir, chunk_size=15):

    image_files = os.listdir(folder_dir)

    return len(image_files)//chunk_size


#Code taken from  Aaqib Saeed, University of Twente.
#  Following function - extract_features
# https://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html

# This converts a segment of audio into melspectrogram, MFCC, and logspectrogram features.
#def extract_features(filepath, bands = 60, frames = 41, bitrate=16000):
def extract_features(sound_clip, bands = 60, bitrate=16000):
    #window_size = 512 * (frames - 1)

    #sound_clip = librosa.load(filepath, sr=bitrate)[0]
    #sound_clip = sound_clip[0:bitrate]

    #fft = librosa.stft(sound_clip, n_fft=2048)
    #print(fft.shape)

    #bands = fft.shape[0]

    melspec = librosa.feature.melspectrogram(sound_clip, n_mels = bands)
    n_mfcc = melspec.shape[0]
    mfcc = librosa.feature.mfcc(sound_clip, sr=bitrate, n_mfcc=n_mfcc)
    #print(mfcc.shape)
    logspec = librosa.amplitude_to_db(melspec)
    #logspec = logspec.T.flatten()[:, np.newaxis].T
    logspec = np.asarray(logspec)

    melspec = np.expand_dims(melspec, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    logspec = np.expand_dims(logspec, axis=0)

    # print(melspec.shape)
    # print(mfcc.shape)
    # print(logspec.shape)

    audio_image = np.vstack([melspec, mfcc, logspec])
    audio_image = np.reshape(audio_image, (audio_image.shape[1], audio_image.shape[2], audio_image.shape[0]))

    return audio_image

#extract_features("final_dataset/audio_final/seq_1_channel_1.wav")

def audio_norm(data):

    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

# Get Sound data according to specifications
def loadAudio(file_path, bitrate=16000):
    data = librosa.core.load(file_path, sr=bitrate)[0]


    data = audio_norm(data)
    #print(data.shape)
    return data


# Each chunk size is about a second
def loadAudioChunk(file_path, bitrate=16000, chunk_index=0, chunk_size = 1):

    allAudio = loadAudio(file_path, bitrate=bitrate)

    startIndex = chunk_index*chunk_size*bitrate
    endIndex = startIndex + chunk_size*bitrate

    assert(endIndex <= allAudio.shape[0]) #Check if we still have enough data

    audio_segment = allAudio[startIndex : endIndex]
    audio_image = extract_features(audio_segment)

    return audio_segment, audio_image



# Get IMU data according to specifications
def loadIMUChunk(filepath, samplingfreq = 10, chunk_index=0, chunk_size=1):

    data = np.loadtxt(filepath, delimiter=',')

    startIndex = chunk_index*chunk_size*samplingfreq
    endIndex = startIndex + chunk_size*samplingfreq

    assert(endIndex <= data.shape[0]) #Check if we still have enough data

    return data[startIndex : endIndex, 1:-1]


# Save the ndarray to a file.
def saveArrayToFile(data, filepath):

    np.save(filepath, data)

# Load the ndarray from a file
def loadArrayFromFile(filepath):
    chunk = np.load(filepath)
    chunk = np.expand_dims(chunk, axis=0)
    return chunk

# Save the contents of a dictonary to file
def saveDictionaryToFile(data, filepath):
    output = open(filepath, "wb")
    pickle.dump(data, output)
    output.close()

# Load the contents of a dictionary from file
def loadDictionaryFromFile(filepath):
    infile = open(filepath, "rb")
    data = pickle.load(infile)
    infile.close()

    return data
