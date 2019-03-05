#import torch
import os
import subprocess


vid_directory = "./video"

input_files = os.listdir(vid_directory)

subprocess.call('mkdir images', shell=True)  #Make the images directory
subprocess.call('mkdir sound', shell=True)  #Make the images directory

for input_file in input_files:
    video_path = os.path.join(vid_directory, input_file)

    name_list = input_file.split("seq")
    current_label = name_list[0]  #Get the current labels
    current_seqnum = name_list[1].split(".")[0]  #Get the current sequence


    if os.path.exists(video_path):
        print(video_path)

        #Create some directories for holding the image and sound data
        output_directory_images = "images/" + current_label
        output_directory_sound = "sound/" + current_label
        if not os.path.exists(output_directory_images):
            subprocess.call('mkdir ' + output_directory_images, shell=True)
        if not os.path.exists(output_directory_sound):
            subprocess.call('mkdir ' + output_directory_sound, shell=True)

        #Pull the images from the video
        # This command also lowers the resolution of the video to be 640 x 360
        create_image_command = 'ffmpeg -i {} -filter:v scale=-1:360 '.format(video_path) + output_directory_images + '/seq_' + current_seqnum + '_image_%05d.jpg'
        subprocess.call(create_image_command,
                        shell=True)

        #Pull the sound from the video, outputs in raw format
        # https://stackoverflow.com/questions/4854513/can-ffmpeg-convert-audio-to-raw-pcm-if-so-how/4854627
        #  Use "play -t raw -r 48k -e signed -b 16 -c 2 seq_01_audio.raw" to check the audio formatting.  This should be correct.
        #create_sound_command = 'ffmpeg -i {} -f s16le -acodec pcm_s16le '.format(video_path) + output_directory_sound + '/seq_' + current_seqnum + '_audio.raw'
        create_sound_command = 'ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 '.format(video_path) + output_directory_sound + '/seq_' + current_seqnum + '_audio.wav'
        subprocess.call(create_sound_command,
                        shell=True)

        #result = classify_video('tmp', input_file, class_names, model, opt)
        #outputs.append(result)

        #subprocess.call('rm -rf tmp', shell=True)
    else:
        print('{} does not exist'.format(input_file))

    #if os.path.exists('tmp'):
#        subprocess.call('rm -rf tmp', shell=True)
