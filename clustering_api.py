import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from datetime import datetime

from sklearn.decomposition import PCA

import pathlib, os, pickle, time, base64, io,sys

#from cuml.cluster import hdbscan
import hdbscan
import uvicorn

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



def remove_file(path: str) -> None:
    os.unlink(path)

def load_pickle(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_features = pickle.load(handle)
    return batch_features.reshape(-1, 4096)

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

def do_hdbscan_cluster(principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: int = 7,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
    global features
    start = time.perf_counter()
    print("starting pca")

    pca = PCA(n_components=principle_components, random_state=random_state)
    pca.fit(features)
    x = pca.transform(features)
    
    clusterer = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)
    start_cluster = time.perf_counter()
    print("starting clusterer")
    clusterer.fit(x)
    start_plot = time.perf_counter()
    print("plotting")
    unique_labels = np.unique(clusterer.labels_)

    # Create a colormap with the same number of colors as the unique labels
    cmap = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))

    fig, ax = plt.subplots()

    sc = ax.scatter(x[:, 0], x[:, 1], c=clusterer.labels_, cmap=cmap)

    # Create a legend using the unique labels and corresponding colors from the colormap
    handles = [plt.Line2D([], [], linestyle='', marker='o', markersize=2, color=cmap(i), label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    ax.legend(handles=handles)

    ax.set_title("HDBSCAN Clustering")
    print("returning")
    fin = time.perf_counter()
    print(f"Timings:\n tot:  {fin- start:0.4f}\nPCA:  {start_cluster-start}\nfit:  {start_plot-start_cluster}\nplot: {fin-start_plot}")
    return fig, ax




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

@app.post('/customhdbscan')
async def custom_hdb(background_tasks: BackgroundTasks):
    global features
    
    
    fig, ax = do_hdbscan_cluster()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    background_tasks.add_task(remove_file, path)
    return JSONResponse(content={"image": img_base64})



features = init(pathlib.Path("features/"))

if __name__ == '__main__':
  uvicorn.run("clustering_api:app", host="0.0.0.0", port=8080, timeout_keep_alive=120)
  