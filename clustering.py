import numpy as np
from tqdm import tqdm
import os
import image_processor

from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.applications.vgg16 import preprocess_input 
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

def cluster(rootFolder : str = "/", principal_components : int = 2, clusters : int = 2):
    model_ft = VGG16()
    model_ft = Model(inputs= model_ft.inputs,outputs = model_ft.layers[-2].output)

    images, filenames = image_processor.processFolder(rootFolder)

    
    x = preprocess_input(images)

    features = model_ft.predict(x, use_multiprocessing=True, verbose=0)

    features = features.reshape(-1,4096)
    pca = PCA(n_components=principal_components, random_state=22)
    pca.fit(features)
    x = pca.transform(features)

    kmeans = KMeans(n_clusters=clusters, random_state=22,n_init="auto")
    kmeans.fit(x)
    
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups