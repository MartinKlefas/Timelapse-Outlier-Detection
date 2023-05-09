import pyarrow.feather as feather
from concurrent.futures import ThreadPoolExecutor
import pathlib, os, pickle, time, base64, uuid
import numpy as np
from tqdm import tqdm
import itertools


import random
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd
import faiss , sys
import gc


def load_pickle(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_features = pickle.load(handle)
    return batch_features.reshape(-1, 4096)

def load_pickle_filesList(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_elems = pickle.load(handle)
    return batch_elems
pickleFolder = pathlib.Path("faiss-test/600k")
feat_pickles = pickleFolder.rglob('features*.pickle')
list_of_pickles =  [str(p) for p in feat_pickles]

with ThreadPoolExecutor() as executor:
    features_list = list(tqdm(executor.map(load_pickle, list_of_pickles), total=len(list_of_pickles), desc="Loading Pickles"))

print(f"concatenating features")
six_k_features = np.concatenate(features_list, axis=0)

print("Reading FileNames")
feat_pickles = pickleFolder.rglob('filenames*.pickle')
list_of_pickles =  [str(p) for p in feat_pickles]

with ThreadPoolExecutor() as executor:
    six_k_filesList = list(itertools.chain.from_iterable(list(executor.map(load_pickle_filesList, list_of_pickles))))




pickleFolder = pathlib.Path("faiss-test/100k")
feat_pickles = pickleFolder.rglob('features*.pickle')
list_of_pickles =  [str(p) for p in feat_pickles]

with ThreadPoolExecutor() as executor:
    features_list = list(tqdm(executor.map(load_pickle, list_of_pickles), total=len(list_of_pickles), desc="Loading Pickles"))

print(f"concatenating features 2")
one_k_features = np.concatenate(features_list, axis=0)


print("Reading FileNames 2")
feat_pickles = pickleFolder.rglob('filenames*.pickle')
list_of_pickles =  [str(p) for p in feat_pickles]

with ThreadPoolExecutor() as executor:
    one_k_filesList = list(itertools.chain.from_iterable(list(executor.map(load_pickle_filesList, list_of_pickles))))

print(f"six hundred features {six_k_features.shape}")
print(f"one hundred Features {one_k_features.shape}")


#change the individual rows into numpy arrays of type float32
db_vectors = [np.array(x, dtype="float32") for x in six_k_features ]

#change it from a list of arrays to an array of arrays.
db_vectors = np.array(db_vectors)

dimension = len(db_vectors[0])    # dimensions of each vector                         
n = len(db_vectors)    # number of vectors  

print(dimension,n)

from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

print("Current Time =", current_time)

print("trying to allocate to GPU:")

res = faiss.StandardGpuResources()

nlist = int(9)  # number of clusters (see above image!)
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)

gpu_index_flat.train(db_vectors)


batch_size = 2000
num_batches = int(np.ceil(n / batch_size))

for i in tqdm(range(num_batches),desc="Processing image batches"):
           

        start = i * batch_size
        end = (i + 1) * batch_size
        batch_vectors = db_vectors[start:end]
        try:
            gpu_index_flat.add(batch_vectors)
        except:
            break


            



print(f"Index created, {gpu_index_flat.ntotal} vectors added. dropping old variables and starting search")
del db_vectors, six_k_features, index
gc.collect()

#change the individual rows into numpy arrays of type float32
search_vectors = [np.array(x, dtype="float32") for x in one_k_features]

#change it from a list of arrays to an array of arrays.
search_vectors = np.array(search_vectors)

distances, indices = gpu_index_flat.search(x=search_vectors,k=1)

with open("distance.pickle","wb") as pickleFile:
        pickle.dump(distances,pickleFile,protocol=pickle.HIGHEST_PROTOCOL)

with open("indices.pickle","wb") as pickleFile:
        pickle.dump(indices,pickleFile,protocol=pickle.HIGHEST_PROTOCOL)

print("done with faiss")

with open("faiss-test/groupPickle.pickle","rb") as pickleFile:
        groups = pickle.load(pickleFile)

# this is one group per cluster with filenames as the value of the sub-list. we want this the other way round, to lookup target group from filename:
encodings = {file: group for group in groups for file in groups[group]}
#encodings is now a dictionary with the key as the filename and the value as the group as predicted by clustering.
successes = 0 
fails = 0
notfound = 0
failDetails = []

for i, idx in enumerate(indices.tolist()):
        try:
                matched_filename = six_k_filesList[idx[0]]
                predicted_group = encodings[matched_filename]
                actual_group = encodings[one_k_filesList[i]]
                if predicted_group == actual_group:
                        successes += 1
                else:
                        fails += 1
                        failDetails.append({"filename":one_k_filesList[i],"matched filename":matched_filename,"predicted group":predicted_group,
                                        "actual group":actual_group,"distance":distances[i]})
        except:
                notfound +=1

print(f"{fails+successes+notfound} matches checkd, {successes *100 / (fails+successes+notfound):.1f} % matches successful")


