import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from concurrent.futures import ThreadPoolExecutor
from itertools import islice



from sklearn.decomposition import PCA

import pathlib, gc, pickle, sys, uuid

import hdbscan

features = []

def load_pickle(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_features = pickle.load(handle)
    return batch_features.reshape(-1, 4096)

def init(pickleFolder: pathlib.Path):
    global features
    print("Reading Features")

    feat_pickles = pickleFolder.rglob('features*.pickle')
    list_of_pickles =  [str(p) for p in feat_pickles]

    with ThreadPoolExecutor() as executor:
        features_list = list(tqdm(executor.map(load_pickle, list_of_pickles), total=len(list_of_pickles), desc="Loading Pickles"))

    print(f"concatenating features, {type(features)}")
    features = np.concatenate(features_list, axis=0)

def do_hdbscan_cluster(principle_components : int = 2, random_state: int =22, alpha:float=1.4, approx_min_span_tree:bool=True,
                gen_min_span_tree:bool=False, leaf_size: int = 40, cluster_selection_epsilon: int = 7,
                metric : str ='manhattan', min_cluster_size:int=5,allow_single_cluster:bool=True):
    global features

    pca = PCA(n_components=principle_components, random_state=random_state)
    pca.fit(features)
    x = pca.transform(features)
    
    clusterer = hdbscan.HDBSCAN( alpha=alpha, approx_min_span_tree=approx_min_span_tree,
    gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, cluster_selection_epsilon=cluster_selection_epsilon,
    metric=metric, min_cluster_size=min_cluster_size, min_samples=None, p=None,allow_single_cluster=allow_single_cluster)

    clusterer.fit(x)

    
    unique_labels = np.unique(clusterer.labels_)

    # Create a colormap with the same number of colors as the unique labels
    cmap = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))

    fig, ax = plt.subplots()

    sc = ax.scatter(x[:, 0], x[:, 1], c=clusterer.labels_, cmap=cmap)

    # Create a legend using the unique labels and corresponding colors from the colormap
    handles = [plt.Line2D([], [], linestyle='', marker='o', markersize=8, color=cmap(i), label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    ax.legend(handles=handles)

    ax.set_title("HDBSCAN Clustering")

    return fig, ax


init(pathlib.Path("Y:\Allotment Timelapse"))

fig, ax = do_hdbscan_cluster()

plt.show()