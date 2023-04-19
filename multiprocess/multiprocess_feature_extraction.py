from multiprocessing import Process, Queue
import os, time
import numpy as np

import pathlib, gc, pickle, sys, uuid

from itertools import islice

import initialise, file_move,pre_process

imagePath = pathlib.Path("Y:/Allotment Timelapse")
forceBar = initialise.diskTesting(imagePath)
result_queue = Queue()

fastCacheFolder = pathlib.Path("cache/")

for directory in [x for x in imagePath.iterdir() if x.is_dir()]:
    print(f"testing {directory} for changes.")
    
    if pathlib.Path(directory / "features_all.pickle").exists() and not initialise.feat_current(directory):
          print("Everything seems to be up to date. Bye!")
          sys.exit()

    new_pickles_folder, model_ft, files = initialise.setup_vars(root_folder=directory, imagePattern="*.jpg")

    doneFilenames, batch_size = initialise.get_progress(root_folder=directory, imagePattern="*.jpg",forceBar=forceBar)

    if forceBar:
         # we again need to use our ugly hack to do a progress bar, this time though we need a list of strings for cv2 to be able to open them
        fileNames = [str(p) for p in files]
        num_batches = int(np.ceil(len(fileNames) / batch_size))
    else:
         num_batches = 1e20 # we don't know how many files there are, so we don't know how many batches there will be.

   
    
    keepGoing = True
    i=0
    while keepGoing: #tqdm(range(num_batches),desc="Processing image batches"):
            if num_batches != 1e20:
                print(f"Batch {i+1} of {num_batches+1}")
            else:
                print(f"Batch {i+1}")

            start = i * batch_size
            end = (i + 1) * batch_size
            
            if forceBar:
                batch_fileNames = fileNames[start:end]
            else:
                batch_fileNames = [str(p) for p in islice(files, batch_size)]

            #initialise threads
            mover = Process(target=file_move.move_batch, args=(batch_fileNames,fastCacheFolder))

            preprocessor = Process(target=pre_process.pre_process, args=(fastCacheFolder,result_queue))

            extractor = Process(target=f, args=('bob',))    
            
            #start jobs
            #join
            #write outputs