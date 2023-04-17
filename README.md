
# Time-lapse Outlier Detection

This project aims to use Machine Vision, unsupervised learning, and clustering techniques to identify outliers in large image datasets, such as those used for time-lapse video creation. The project is a work in progress and has been undertaken in stages to provide a comprehensive understanding of the development process.
  

## Stage 1: Proof of Concept

  
I began by creating a simple case: a single image of a child's toy on a white background, decomposed into 100 sub-images. Using k-means clustering in[image_clustering.ipynb](image_clustering.ipynb), I implemented a pre-trained CNN called "VGG16" to extract key features from each image and group them based on similarity.

The clusters where then translated into groups of files for visualisation - correctly showing that I had created a group of background images and a group with just the images of the toy.

  

I refined the process in [image_clustering_innate_parallel.ipynb](https://chat.openai.com/c/image_clustering_innate_parallel.ipynb) by importing all images at once and passing them to the encoder in a batch, speeding up the test case.

  

## Stage 2: HDBSCAN Implementation

For real data, I don't know the number of clusters beforehand, so I used an [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html) implementation in [hdbscan_clustering.ipynb](hdbscan_clustering.ipynb) Using the same feature extraction technique, the HDBSCAN produced comparable results on the test dataset, grouping background images into a cluster and leaving images with the toy in the "noise" group, as shown below:

![clustered plot of encoded image segments](test_hdbscan.png)

  


## Stage 3: Large-Scale Test Data

I implemented two approaches to handle large-scale test data:

1.  **Frequent saving:** In [batchwise_feature_extraction.py](batchwise_feature_extraction.py), I divided the input dataset into batches sized to fit into 8-10GB of VRAM and saved the resulting encoding as a pickle file after each batch.
    
2.  **Parallel Workers:** I designed a parallel worker implementation to minimize downtime for any resource, which is still under development. The logic for this implementation is shown below

  

## Stage 4: Clustering with API

With a large number of realistic images now encoded, I implemented clustering based on Stage 2 in [hdb_clustering.py](hdb_clustering.py) and then [clustering_api.py](clustering_api.py), enabling it to run under uvicorn and RapidAPI. Users can tIak parameters, and the feature encodings can be updated over time.

  

This code is still under development, but at present the clustering process itself takes too long for many API callers, and thus it has been implemented as a background process. The user request is first sorted into ones that have been computed already (either as standard or by another user) and then either known results are returned, or the result is computed in the backend to be returned next time. This is a bit of a hacky method to get around the lack of GPU compatible clustering code for windows machines, and the lack of budget to rent a GPU enabled linux machine from a cloud provider. In a real implementation I'd be using [cUML Clustering from Rapids.ai](https://developer.nvidia.com/blog/faster-hdbscan-soft-clustering-with-rapids-cuml/) to run the clustering in milliseconds rather than minutes.

  


## Stage 5: Interactivity and Visualization

In this stage (not yet started), I plan to add interactivity to the plots and allow users to visualize image groups, interact with the plot, and potentially select subsets of data to re-cluster or manipulate. Users will be able to request a list of files in a particular cluster for further processing outside of this package.

  
  

> Written with [StackEdit](https://stackedit.io/).