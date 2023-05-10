from fastapi import File, FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from typing import List
from typing_extensions import Annotated

from concurrent.futures import ThreadPoolExecutor
import pathlib, pickle
import numpy as np
import itertools

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 


import numpy as np 
import faiss 
import zlib

import uvicorn, random
from datetime import datetime

# my modules
import kmeans_preprocessor


if __name__ == '__main__':
    print("Starting server")
    uvicorn.run("faiss-api:app", host="0.0.0.0", port=8080, timeout_keep_alive=120)
else:
    start_time = datetime.now()
    indexPath = "faiss-test/index.index"
    

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/healthcheck')
def healthcheck():
    timediff = datetime.now() - start_time

    days = timediff.days
    seconds = timediff.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    msg = "API has been running for "
    if days > 0:
        msg += f"{days} days, "
    if hours > 0:
        msg += f"{hours} hours, "
    try:  
        temp_index = faiss.read_index(indexPath)
        format_num_images = format(temp_index.ntotal,",")
    except:
        return JSONResponse(content={"Error":f"FAISS index file {indexPath} not found, exiting" })

    msg += f"{minutes:02d}:{seconds:02d}, and is searcing on {format_num_images} images."

  
    return {"message": msg}

@app.post("ImagesToGroupIDs")
async def predict_image_groups(files : Annotated[List[UploadFile], File(description="One or more image files as UploadFile")], useZip : Annotated[str, Form("Should the results be zlib compressed when returned (t/f - default is no)?")]):
    useZip = useZip.lower() in ['true', '1', 't', 'y', 'yes', 'on']

    if pathlib.Path(indexPath).exists():
        if len(files) < 2000:


            batch_fileNames = [file.filename for file in files]
            with ThreadPoolExecutor() as executor:
                        images = list(executor.map(kmeans_preprocessor.cvload_image, batch_fileNames, [224]*len(batch_fileNames)))
            
            images = [img for img in images if img is not None]
        
            images = np.concatenate(images, axis=0)

            x = preprocess_input(images)
            del images
            model_ft = VGG16(weights="imagenet")
            model_ft = Model(inputs= model_ft.inputs,outputs = model_ft.layers[-2].output)
            batch_features = model_ft.predict(x, use_multiprocessing=True, verbose=0)

            try:  
                res = faiss.StandardGpuResources()  
                gpu_index_flat = faiss.index_cpu_to_gpu(res, 0,faiss.read_index(indexPath))
                print(f"Index loaded, {gpu_index_flat.ntotal} vectors present.")
                search_vectors = [np.array(x, dtype="float32") for x in batch_features]
                search_vectors = np.array(search_vectors)

                distances, indices = gpu_index_flat.search(x=search_vectors,k=1)
                predicted_group = []
                for i, idx in enumerate(indices.tolist()):
                    try:
                        with open("faiss-test/groupPickle.pickle","rb") as pickleFile:
                            groups = pickle.load(pickleFile)
                        matched_filename = getMatchedFileName[idx[0]]
                        encodings = {file: group for group in groups for file in groups[group]}
                        predicted_group.append({"filename": batch_fileNames[i],"predicted group": encodings[matched_filename],"distance":distances[i]})

                    except:
                        continue


                if not useZip:
                    return JSONResponse({"message":"Success","zip":False,"Predictions":predicted_group})
                else:
                    predicted_group_zipped = zlib.compress(predicted_group.encode())
                    return JSONResponse({"message":"Success","zip":True,"Predictions":predicted_group_zipped})
                
            except:
                return JSONResponse(content={"Error":f"FAISS index file {indexPath} either failed to load or was not searchable" })




        else:
            return JSONResponse(content={"Error":f"Seriously, maybe {len(files)} is too many. Please restrict uploads to 2000 next time" })
    else:
        return JSONResponse(content={"Error":f"FAISS index file {indexPath} not found, exiting" })

def load_pickle_filesList(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_elems = pickle.load(handle)
    return batch_elems

def getMatchedFileName(idx : int):
    pickleFolder = pathlib.Path("faiss-test/600k")
    file_pickles = pickleFolder.rglob('filenames*.pickle')
    list_of_pickles =  [str(p) for p in file_pickles]
    with ThreadPoolExecutor() as executor:
        six_k_filesList = list(itertools.chain.from_iterable(list(executor.map(load_pickle_filesList, list_of_pickles))))

    return six_k_filesList[idx[0]]