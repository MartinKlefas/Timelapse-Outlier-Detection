# Time-lapse Outlier Detection

This is a practice project to use Machine Vision, unsupervised learning and clustering to find outliers in large image datasets, such as those intended for time-lapse video creation. It is a work in progress, and has been undertaken in stages, with each retained to better explain the process by which the final project was created.

# Stage 1

We first need a very simple case to act as a proof of concept, with a known intended outcome. In this instance we have a single image of a child's toy on a white background. We then decomposed this image into 100 sub-images so that most of them were mostly white, and some would have all or part of the toy in them.

As this is a simple case, and we know we want 2 clusters, we then implemented k-means clustering in [image_clustering.ipynb](image_clustering.ipynb). In this implementation we used a pre-trained CNN known as "VGG16" to extract key features of each image in the set, and then clustered these in two dimensions to group them by their broad similarity.

The clusters where then translated into groups of files for visualisation - correctly showing that we had created a group of background images and a group with just the images of the toy.

This was further refined in [image_clustering_innate_parallel.ipynb](image_clustering_innate_parallel.ipynb) to import all the images at once and pass them all to the encoder to classify as a batch. This just reduced the number of initialisations and calls in the process, speeding everything up for our test case.

# Stage 2
In the real data we won't know ahead of time how many clusters of data there are - there may be one for "night" one for "empty road" one for "lots of cars" one for "that bird that likes to sit on the camera" etc. This means we can't use a K-Means clusterer as it requires the target number of clusters. We're also looking for images that perhaps are completely unique, which wouldn't really belong to any clusters. This kind of data is best classified by a tool which can run something like a proximity search or density search, whereby it inspects the data and finds clusters and separations between clusters in a self-guided fashion.

This in mind an [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html)  implementation was created in [hdbscan_clustering.ipynb](hdbscan_clustering.ipynb) This used the same feature extraction technique, and produced comparible results on the test dataset - grouping all of the background images into a cluster, and leaving the remainder of images with the toy or other coloured artifact in the "noise" group, as shown below:
![clustered plot of encoded image segments](test_hdbscan.png)

# Stage 3
Now that we have working clustering implementations, we need to get some more representative test data to measure them against. Part of the challenge of this is that the potential dataset is larger than the fast local storage available at a reasonable cost for a test project. Two approaches to this problem have been taken:
1. **Frequent saving:** The feature extraction code was moved into a separate implementation, based on the innate parallel code, but deviding the input dataset into batches. The batches are sized to fit into 8-10GB of VRAM and run sequentially. After each batch has been modelled, the resulting encoding is saved out to a pickle file for later processing. This code is implemented in [batchwise_feature_extraction.py](batchwise_feature_extraction.py)
2. **Parallel Workers** The frequent saving code still processes the input data sequentially, meaning that while the data is being read from the slow source repository the GPU sits idle, and while the GPU is crunching the model the data repository gets a bit of a rest. A parallel worker implementation was designed such that any downtime for any resource is minimised. This code is under development

# Stage 4
With a large number of realistic images now encoded, a method for clustering these features was implemented based on the work in stage 2. The clustering was implemented in [hdb_clustering.py](hdb_clustering.py) and then migrated over to [clustering_api.py](clustering_api.py) which has been implemented such that it can be run under uvicorn & rapidAPI as an API, allowing the user to tweak various parameters and for the feature encodings to be updated over time as new images are created and encoded.

This code is still under development, but at present the clustering process itself takes too long for many API callers, and thus it has been implemented as a background process. The user request is first sorted into clusterings that have been computed already (either as standard or by another user) and then either known results are returned, or the result is computed in the backend to be returned next time. This is a bit of a hacky method to get around the lack of GPU compatible clustering code for windows machines, and the lack of budget to rent a GPU enabled linux machine from a cloud provider.  In a real implementation we'd be using [cUML Clustering from Rapids.ai](https://developer.nvidia.com/blog/faster-hdbscan-soft-clustering-with-rapids-cuml/) to run the clustering in milliseconds rather than minutes.

# Stage 5
This has not yet been started, but the intention is to provide some interactivity to the plots, allowing the user to visualise image groups, interact with the plot and potentially select sub-sets of the data to re-cluster or manipulate.

It would also be beneficial to allow the user to request a list of files in a particular cluster, such that they can be removed from the timelapse or processed in some other way outside of this package.


> Written with [StackEdit](https://stackedit.io/).