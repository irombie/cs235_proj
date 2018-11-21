import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
from PIL import Image


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()

def draw_spectogram(path_to_audio, save_path):
    data, sampling_rate = librosa.load(path_to_audio) #os.path.join(root, directory, file_))
    specs = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(specs,ref=np.max), y_axis='mel', fmax=9000,x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

''' make 3 pics from 1 pic
root_to_walk = '/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres_cropped'
root_to_save = '/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres_tripled'
for root, dirs, files in os.walk(root_to_walk):
    for directory in dirs:
        for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
            for f in files1:
                if not os.path.exists(os.path.join(root_to_save,directory)):
                    os.makedirs(os.path.join(root_to_save,directory))
                print("Doing {}".format(os.path.join(root1,f)))
                crop(os.path.join(root1, f), (0, 0, 300, 324), "{}/{}_1.png".format(os.path.join(root_to_save, directory),f))
                crop(os.path.join(root1, f), (300, 0, 600, 324), "{}/{}_2.png".format(os.path.join(root_to_save, directory),f))
                crop(os.path.join(root1, f), (600, 0, 900, 324), "{}/{}_3.png".format(os.path.join(root_to_save, directory),f))
'''
                
''' crop 3 pixels to make calculations easier
root_to_walk = '/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres'
root_to_save = '/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres_cropped'
for root, dirs, files in os.walk(root_to_walk):
    for directory in dirs:
        for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
            for f in files1:
                if not os.path.exists(os.path.join(root_to_save, directory)):
                    os.makedirs(os.path.join(root_to_save, directory)
                crop(os.path.join(root1, f), (0, 0, 900, 324), "{}/{}.png".format(os.path.join(root_to_save, directory),f)
    path_to_save = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig("{}/{}.png".format(save_path,f))
    
    crop("{}/{}.png".format(path_to_save,f), (80,19,983,343),"{}/{}.png".format(path_to_save,f))
   

path = '/Users/iremergun/Downloads/genres'
root_to_save = '/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres'
for root, dirs, files in os.walk(path):
    for directory in dirs:
        for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
            for f in files1:
                print(os.path.join(root1, f))
                draw_spectogram(os.path.join(root1, f), os.path.join(root_to_save, directory))

 '''

