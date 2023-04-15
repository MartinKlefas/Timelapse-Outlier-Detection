from tqdm import tqdm
import numpy as np
import cv2

from concurrent.futures import ThreadPoolExecutor


from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 

import pathlib, gc

defaultSize = 224

def validate_output_dir(tPath : str):
    try:
        pathlib.Path(tPath).mkdir(parents=True, exist_ok=True)
        return True, None
    except Exception as ex:

        return False, ex

def processImage(thisPath : str = "" , thisBuffer = None):
       
    if thisPath:
        myImage = load_img(thisPath, target_size=(defaultSize,defaultSize))
        return preprocess_input(np.array(myImage).reshape(1,defaultSize,defaultSize,3))
    
    if thisBuffer:
        myImage = load_img(thisBuffer, target_size=(defaultSize,defaultSize))
        return preprocess_input(np.array(myImage).reshape(1,defaultSize,defaultSize,3))




def processFolder(objPath : pathlib.Path = None, strPath: str ="", size: int = defaultSize):
    if strPath:
        if not strPath[:-1] =="/":
            strPath = strPath + "/"

        objPath = pathlib.Path(strPath)

    
    print(f"importing images in {objPath}")
    
    
    new_images = np.empty((0, size, size, 3))

    files = objPath.rglob('*s.png')

    # we again need to use our ugly hack to do a progress bar, this time though we need a list of strings for cv2 to be able to open them
    filesList = [str(p) for p in files]
    # we now have a list of definite size, instead of a generator
    print(f"Found {len(filesList)} files.")
    
    print("trimming for test")
    filesList = filesList[:4000]

    with ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(cvload_image, filesList, [size]*len(filesList)),
                       total=len(filesList),
                       bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]',
                       desc="Importing"))

    print("Concatenating")
    new_images = np.concatenate(images, axis=0)
    del images

    gc.collect()
    
    return new_images , filesList
            


def cvload_image(file, size):
  try:
    img = cv2.imread(file)
    if img is None:
            return None
    img = cv2.resize(img, (size, size))
    return img.reshape(1, size, size, 3)
  except Exception as e:
    return None 


def fullPreProcess_Image(filePath: str):
    image = processImage(thisPath= filePath)

    return image


    

def getEmbedding(image, model):


    keImage = fullPreProcess_Image(filePath=image)


    my_embedding = model.predict(keImage, use_multiprocessing=True, verbose=0)



    return my_embedding

def groups_need_updating(folder : pathlib.Path):
    files = folder.rglob('*s.png')
    latest_file = max(files, key=lambda p: p.lstat().st_mtime)
    newest_file_time = latest_file.lstat().st_mtime
    pickle_file_time = pathlib.Path(folder / "groups.pickle").lstat().st_mtime

    return pickle_file_time < newest_file_time
