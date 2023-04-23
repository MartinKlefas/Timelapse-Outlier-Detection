import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import mpld3

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pyarrow.feather as feather

from pydantic import BaseModel

from datetime import datetime

from sklearn.decomposition import PCA

import pathlib, os, pickle, time, base64, uuid,sys, itertools

from cuml.cluster import hdbscan
#import hdbscan
import uvicorn
import pandas as pd

import zlib

from fastapi import File
from fastapi import UploadFile
from fastapi import Form
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

from starlette.background import BackgroundTasks


import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi.middleware.cors import CORSMiddleware

class HDBSCANParams(BaseModel):
    principle_components: int = 2
    random_state: int = 22
    alpha: float = 1.4
    approx_min_span_tree: bool = True
    gen_min_span_tree: bool = False
    leaf_size: int = 40
    cluster_selection_epsilon: float = 1.0
    metric: str = 'manhattan'
    min_cluster_size: int = 5
    allow_single_cluster: bool = True

def remove_file(path: str) -> None:
    os.unlink(path)

def load_pickle(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_features = pickle.load(handle)
    return batch_features.reshape(-1, 4096)
def load_pickle_list(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_elems = pickle.load(handle)
    return batch_elems

def init(pickleFolder: pathlib.Path):
    print("Reading Features")

    feat_pickles = pickleFolder.rglob('features*.pickle')
    list_of_pickles =  [str(p) for p in feat_pickles]

    with ThreadPoolExecutor() as executor:
        features_list = list(tqdm(executor.map(load_pickle, list_of_pickles), total=len(list_of_pickles), desc="Loading Pickles"))

    print(f"concatenating features")
    features = np.concatenate(features_list, axis=0)
    print(f"{features.shape[0]} features loaded")

    
    return features

def readFileNames(pickleFolder: pathlib.Path):
    print("Reading FileNames")

    feat_pickles = pickleFolder.rglob('filenames*.pickle')
    list_of_pickles =  [str(p) for p in feat_pickles]

    with ThreadPoolExecutor() as executor:
        filesList = list(itertools.chain.from_iterable(list(executor.map(load_pickle_list, list_of_pickles))))
    
    return filesList

def do_hdbscan_cluster(principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: float = 1.0,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
    global features
    try:
        figures = feather.read_feather("plots/inter_plots.feather")
    except:
        figures = None

    #start = time.perf_counter()
    print("starting pca")

    pca = PCA(n_components=principle_components, random_state=random_state)
    pca.fit(features)
    x = pca.transform(features)
    
    clusterer = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)
    #start_cluster = time.perf_counter()
    print("starting clusterer")
    clusterer.fit(x)

    print("plotting")
    unique_labels = np.unique(clusterer.labels_)

    # Create a colormap with the same number of colors as the unique labels
    cmap = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))

    
    if principle_components <= 2:
        fig, (ax, lax) = plt.subplots(ncols=2,gridspec_kw={"width_ratios":[4, 1]})
        sc = ax.scatter(x[:, 0], x[:, 1], c=clusterer.labels_, cmap=cmap)
    else:
        fig, (ax, lax) = plt.subplots(ncols=2, subplot_kw={'projection': "3d"},gridspec_kw={"width_ratios":[4, 1]})
        sc = ax.scatter(x[:, 0], x[:, 1],x[:,2], c=clusterer.labels_, cmap=cmap)

    # Create a legend using the unique labels and corresponding colors from the colormap
    handles = [plt.Line2D([], [], linestyle='', marker='o', markersize=2, color=cmap(i), label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    lax.legend(handles=handles)
    lax.axis("off")

    

    ax.set_title("HDBSCAN Clustering")
    plt.tight_layout()
    print("returning")

    plot = mpld3.fig_to_html(fig=fig)
    a = zlib.compress(plot.encode())
    d= {"plot":a,"principle_components" : principle_components, "random_state" : random_state, "alpha":alpha, "approx_min_span_tree":approx_min_span_tree,
               "gen_min_span_tree": gen_min_span_tree, "leaf_size" : leaf_size, "cluster_selection_epsilon" : cluster_selection_epsilon,
                "metric": metric, "min_cluster_size":min_cluster_size,"allow_single_cluster":allow_single_cluster}
    df_dictionary = pd.DataFrame([d],index=[0])
    if not (figures is None):
        figures = pd.concat([figures,df_dictionary], ignore_index=True)
    else:
        figures = df_dictionary
    
    feather.write_feather(df=figures,dest="plots/inter_plots.feather")
    

def already_done(principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: float = 1.0,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
    
    try:
        figures = feather.read_feather("plots/inter_plots.feather")
    
        results = figures.loc[(figures['principle_components'] == principle_components) & 
                            (figures['random_state'] == random_state) & 
                            (figures['alpha'] == alpha) & 
                            (figures['approx_min_span_tree'] == approx_min_span_tree) & 
                            (figures['gen_min_span_tree'] == gen_min_span_tree) & 
                            (figures['leaf_size'] == leaf_size) & 
                            (figures['cluster_selection_epsilon'] == cluster_selection_epsilon) & 
                            (figures['metric'] == metric) & 
                            (figures['min_cluster_size'] == min_cluster_size) & 
                            (figures['allow_single_cluster'] == allow_single_cluster)  
                            ,
                    ['plot']]
    except:
        results = None


    return results

def getGroups(filenames,principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: float = 1.0,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
   
    #start = time.perf_counter()
    print("starting pca")

    pca = PCA(n_components=principle_components, random_state=random_state)
    pca.fit(features)
    x = pca.transform(features)
    
    clusterer = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)
    #start_cluster = time.perf_counter()
    print("starting clusterer")
    clusterer.fit(x)    




    groups = {}
    for file, cluster in zip(filenames,clusterer.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups

app = FastAPI()
print("Starting server")
start_time = datetime.now()

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


  msg = f"API has been running for {str(timediff )}, and is clustering info on {features.shape[0]} images."
  
  return {"message": msg}

@app.post('/defaulthdbscan')
def default_hdb():
    

    return FileResponse(path="plots/defaulthdb.png")

def remove_file(path: str) -> None:
    os.unlink(path)

@app.post('/cHdbscanGrp')
async def custom_hdb(background_tasks: BackgroundTasks,params: HDBSCANParams):
    global fileNames
    groups = getGroups(fileNames, principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    with open("groupPickle.pickle","wb") as pickleFile:
        pickle.dump(groups,pickleFile)
    
    background_tasks.add_task(remove_file, "groupPickle.pickle")

    return FileResponse(path="groupPickle.pickle")


@app.post('/customhdbscan')
async def custom_hdb(background_tasks: BackgroundTasks, params: HDBSCANParams):

    
    found = already_done(principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    if not (found is None) and len(found) > 0:
        print("returning image")
       
        inter_plot = zlib.decompress(found["plot"].iloc[0]).decode()

       

        return JSONResponse(content={"message": "found", "plot": inter_plot})
    else:
        print("adding job")

        background_tasks.add_task(do_hdbscan_cluster,principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    
        return JSONResponse(content={"message": "please wait", "wait_time": "100"})



features = init(pathlib.Path("features/"))
fileNames = readFileNames(pathlib.Path("features/"))

if __name__ == '__main__':
  uvicorn.run("clustering_api_linux_interactive:app", host="0.0.0.0", port=8080, timeout_keep_alive=120)
  