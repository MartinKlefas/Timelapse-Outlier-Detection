import pathlib, gc

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from keras.applications.vgg16 import preprocess_input 
from multiprocessing import Queue

import cv2

def cvload_image(file, size):
  try:
    img = cv2.imread(file)
    if img is None:
            return None
    img = cv2.resize(img, (size, size))
    return img.reshape(1, size, size, 3)
  except Exception as e:
    return None 

def pre_process(cacheFolder: pathlib.Path,imagePattern : str,result_queue: Queue):
    files = cacheFolder.rglob(imagePattern)
    filesList = [str(p) for p in files]
    
    if len(filesList) > 0:
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(cvload_image, filesList, [224]*len(filesList)))

        images = [img for img in images if img is not None]
                    #print("Concatenating")
        images = np.concatenate(images, axis=0)
        x = preprocess_input(images)
        result_queue.put(x)
        del images, x
        gc.collect()

    