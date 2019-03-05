#import torch
import os


vid_directory = "./video"

input_files = os.listdir(vid_directory)
print(input_files)

outputs = []
for input_file in input_files:
    video_path = os.path.join(vid_directory, input_file)
    if os.path.exists(video_path):
        print(video_path)
        subprocess.call('mkdir tmp', shell=True)
        subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
                        shell=True)

        result = classify_video('tmp', input_file, class_names, model, opt)
        outputs.append(result)

        subprocess.call('rm -rf tmp', shell=True)
    else:
        print('{} does not exist'.format(input_file))

    #if os.path.exists('tmp'):
#        subprocess.call('rm -rf tmp', shell=True)
