import numpy as np
from tqdm import tqdm

import kmeans_preprocessor, disk_test

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pathlib, gc, pickle, sys, uuid


def get_yes_no_input(question):
    while True:
        response = input(f"{question}? (Y/n) ").lower().strip()
        if response == '' or response == 'y' or response == 'yes':
            return True
        elif response == 'n' or response == 'no':
            return False
        else:
            print("Invalid input. Please enter 'Y', 'N', or leave empty for default (Y).")


def feat_current(folder : pathlib.Path, imagePattern : str):
    files = folder.rglob(imagePattern)
    latest_file = max(files, key=lambda p: p.lstat().st_mtime)
    newest_file_time = latest_file.lstat().st_mtime
    pickle_file_time = pathlib.Path(folder / "features_all.pickle").lstat().st_mtime

    return pickle_file_time > newest_file_time

def batches_current(folder : pathlib.Path, imagePattern : str):
    files = folder.rglob(imagePattern)
    latest_image = max(files, key=lambda p: p.lstat().st_mtime)

    newest_image_time = latest_image.lstat().st_mtime

    files = folder.rglob('filenames_batch_*.pickle')
    latest_pickle = max(files, key=lambda p: p.lstat().st_mtime)

    newest_pickle_time = latest_pickle.lstat().st_mtime

    return newest_pickle_time > newest_image_time

def diskTesting(rootFolder : pathlib.Path = pathlib.Path("/")):
    diskInfo = disk_test.get_drive_info(Path=rootFolder)

    if diskInfo["InterfaceType"] == "USB":
        print("You appear to have selected a USB drive. Initialising the progress metrics can take a long time for some USB drives.")
        return not get_yes_no_input("Would you like to skip this to speed things up")
    else:
        return True

def setup_vars(rootFolder : pathlib.Path = pathlib.Path("/"), imagePattern : str = "*s.png"):  
    new_pickles_folder = pathlib.Path(rootFolder / "pickles" / str(uuid.uuid4()) / "")
    new_pickles_folder.mkdir(parents=True, exist_ok=True)

    model_ft = VGG16()
    model_ft = Model(inputs= model_ft.inputs,outputs = model_ft.layers[-2].output)
    print("finding files")
    files = rootFolder.rglob(imagePattern)

    return new_pickles_folder, model_ft, files

def get_progress(rootFolder : pathlib.Path = pathlib.Path("/"), forceBar : bool = False,imagePattern : str = "*s.png"):

    filenames_pickles = rootFolder.rglob('filenames_batch_*.pickle')
    list_of_pickles =  [str(p) for p in filenames_pickles] # this should be a relatively short list, so it shouldn't cause the same impact as "listing" the image files from the generator

    if len(list_of_pickles) > 0 and (not forceBar or batches_current(rootFolder,imagePattern)): # the batches_current feature scans all files and folders for any updates after the pickle.
        #this would take a long time on usb/spinning disks/etc so we follow the same rule as the progress bar, hoping it'll be ok.
        print("reading old progress")

        filename_pickles = rootFolder.rglob('filenames*.pickle')
        
        
        for thisPickle in filename_pickles:
            with open(str(thisPickle), 'rb') as handle:
                    batch_fileNames = pickle.load(handle)
            doneFilenames = doneFilenames + batch_fileNames

        #Change list to set, this is because checking membership in a set is much faster (O(1) complexity) than in a list (O(n) complexity)
        doneFilenames = set(doneFilenames)
        print(f"loaded progress for {len(doneFilenames)} files")
        
    batch_size = 2000

    return doneFilenames, batch_size