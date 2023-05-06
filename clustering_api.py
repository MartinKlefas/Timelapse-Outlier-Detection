import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pyarrow.feather as feather

from pydantic import BaseModel

from datetime import datetime

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import pathlib, os, pickle, time, base64, uuid,sys, itertools, gc

from scipy.sparse import vstack, csr_matrix


#from cuml.cluster import hdbscan

import hdbscan
import uvicorn, random
import pandas as pd

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
    return csr_matrix(batch_features.reshape(-1, 4096))

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
    features = vstack(features_list)
    print(f"{features.shape[0]} features loaded")
    del features_list

    gc.collect()
    
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
        figures = feather.read_feather("plots/plots.feather")
    except:
        figures = None

    #start = time.perf_counter()
    print("starting pca")
    
    # migrated to sparse matrices so we need to use TruncatedSVD (PCA for sparse matrices)

    pca = TruncatedSVD(n_components=principle_components, random_state=random_state)
    x =  pca.fit_transform(features)

    
    clusterer = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)
    #start_cluster = time.perf_counter()
    print("starting clusterer")
    clusterer.fit(x)

    del x

    gc.collect()

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
   # fin = time.perf_counter()
   # print(f"Timings:\n tot:  {fin- start:0.4f}\nPCA:  {start_cluster-start}\nfit:  {start_plot-start_cluster}\nplot: {fin-start_plot}")
    filename = "plots/" + str(uuid.uuid4()) + ".png"
    plt.savefig(filename)

    d= {"filename":filename,"principle_components" : principle_components, "random_state" : random_state, "alpha":alpha, "approx_min_span_tree":approx_min_span_tree,
               "gen_min_span_tree": gen_min_span_tree, "leaf_size" : leaf_size, "cluster_selection_epsilon" : cluster_selection_epsilon,
                "metric": metric, "min_cluster_size":min_cluster_size,"allow_single_cluster":allow_single_cluster}
    df_dictionary = pd.DataFrame([d],index=[0])
    if not (figures is None):
        figures = pd.concat([figures,df_dictionary], ignore_index=True)
    else:
        figures = df_dictionary

    feather.write_feather(df=figures,dest="plots/plots.feather")

def already_done(principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: float = 1.0,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
    
    try:
        figures = feather.read_feather("plots/plots.feather")
    
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
                    ['filename']]
    except:
        results = None


    return results

def getGroups(filenames,principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: float = 1.0,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
   
    #start = time.perf_counter()
    print("starting pca")

    # migrated to sparse matrices so we need to use TruncatedSVD (PCA for sparse matrices)
    pca = TruncatedSVD(n_components=principle_components, random_state=random_state)
    x =  pca.fit_transform(features)
    
    clusterer = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)
    #start_cluster = time.perf_counter()
    print("starting clusterer")
    clusterer.fit(x)    

    del x
    gc.collect()

    groups = {}
    for file, cluster in zip(filenames,clusterer.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups

def trimFileName(fullPath : str):
    trimmedPath = fullPath.replace("D:\\Allotment Timelapse\\","")
    trimmedPath = trimmedPath.replace("Y:\\Allotment Timelapse\\GoPro TimeLapse\\","")
    trimmedPath = trimmedPath.replace("C:\\xampp\\htdocs\\clustering\\thumbnails\\All","")
    return trimmedPath

def getSamples(groups, num_samples : int = 10):
    samples = {}
    sizes = {}
    for gNum, files in groups.items():
        sizes[gNum.item()] = len(files)
        if len(files) > num_samples:
            samples[gNum.item()] = [trimFileName(x) for x in random.sample(files,k=num_samples)]
        else:
            samples[gNum.item()] = [trimFileName(x) for x in files]
    
    return samples, sizes

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


    format_num_images = format(features.shape[0],",")

    msg += f"{minutes:02d}:{seconds:02d}, and is clustering info on {format_num_images} images."

  
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
        pickle.dump(groups,pickleFile,protocol=pickle.HIGHEST_PROTOCOL)
    
    #background_tasks.add_task(remove_file, "groupPickle.pickle")

    return FileResponse(path="groupPickle.pickle")

@app.post('/hdbSamples')
async def custom_hdb(background_tasks: BackgroundTasks,params: HDBSCANParams):
    global fileNames
    groups = getGroups(fileNames, principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    with open("groupPickle.pickle","wb") as pickleFile:
        pickle.dump(groups,pickleFile,protocol=pickle.HIGHEST_PROTOCOL)
    

    with open("groupPickle.pickle", "rb") as f:
        file_content = f.read()
    file_base64 = base64.b64encode(file_content).decode("utf-8")
    

    samples,sizes = getSamples(groups,20)



    #background_tasks.add_task(remove_file, "groupPickle.pickle")
   
    return JSONResponse(content={"file_base64": file_base64,"filename": "groupPickle.pickle", "image_groups": samples, "len_groups": sizes})

@app.get('/screeplot')
async def scree_plot():

   
    if not pathlib.Path('scree plot.png').exists():
        print("svd")
        # Apply TruncatedSVD
        n_components = 50
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(features)
        print("variance")
        # Calculate the explained variance for each component
        explained_variance = svd.explained_variance_ratio_

        print("plotting")
        # Create a scree plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='-', label='Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.legend()
        plt.grid()
        plt.savefig('scree plot.png')
        
    return FileResponse('scree plot.png')


@app.post('/customhdbscan')
async def custom_hdb(background_tasks: BackgroundTasks, params: HDBSCANParams):

    
    found = already_done(principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    if not (found is None) and len(found) > 0:
        print("returning image")
       
        with open(found["filename"].to_string(index=False), "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

        return JSONResponse(content={"message": "found", "image": img_base64})
    else:
        print("adding job")

        background_tasks.add_task(do_hdbscan_cluster,principle_components=params.principle_components,random_state=params.random_state,
                                    alpha=params.alpha, approx_min_span_tree=params.approx_min_span_tree,gen_min_span_tree=params.gen_min_span_tree,
                                    leaf_size=params.leaf_size,  cluster_selection_epsilon=params.cluster_selection_epsilon,
                                    metric=params.metric, min_cluster_size=params.min_cluster_size, 
                                    allow_single_cluster=params.allow_single_cluster)
    
        return JSONResponse(content={"message": "please wait", "wait_time": "100"})

if __name__ == '__main__':
    print("Starting server")
    uvicorn.run("clustering_api:app", host="0.0.0.0", port=8080, timeout_keep_alive=120)
else:
    start_time = datetime.now()
    features = init(pathlib.Path("features/"))
    fileNames = readFileNames(pathlib.Path("features/"))
    # probably not relevant anymore
    if pathlib.Path("scree plot.png").exists():
        pathlib.Path("scree plot.png").unlink()
  