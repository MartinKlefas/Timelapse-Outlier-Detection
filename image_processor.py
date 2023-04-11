from tqdm import tqdm
import numpy as np
import os
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 

import pathlib

defaultSize = 224

def validate_output_dir(tPath : str):
    try:
        pathlib.Path(tPath).mkdir(parents=True, exist_ok=True)
        return True, None
    except Exception as ex:

        return False, ex

def processImage(thisPath : str = "" , thisBuffer = None):
       
    if thisPath:
        myImage = load_img(thisPath, target_size=(224,224))
        return preprocess_input(np.array(myImage).reshape(1,224,224,3))
    
    if thisBuffer:
        myImage = load_img(thisBuffer, target_size=(224,224))
        return preprocess_input(np.array(myImage).reshape(1,224,224,3))
    
def processFolder(path: str, size: int = defaultSize):
    if not path[:-1] =="/":
        path = path + "/"

    objPath = pathlib.Path(path)

    
    
    new_images = np.empty((0, size, size, 3))
    filenames = []
    files = objPath.glob('*.[jpg][png][jpeg][tiff]')
    for item in tqdm(files,bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]',desc="Importing"):
        new_images =  np.concatenate((new_images,np.array(load_img(item , target_size=(size,size))).reshape(1,size,size,3)), axis = 0)
    
    return new_images , files
            





def fullPreProcess_Image(filePath: str):
    image = processImage(thisPath= filePath)

    return image


    

def getEmbedding(image, model):


    keImage = fullPreProcess_Image(filePath=image)


    my_embedding = model.predict(keImage, use_multiprocessing=True, verbose=0)



    return my_embedding
