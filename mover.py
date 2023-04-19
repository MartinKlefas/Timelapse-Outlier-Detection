import kmeans_clustering

import pyarrow.feather as feather

import sys,pathlib, random,shutil


groups = kmeans_clustering.cluster(rootFolder = sys.argv[1:], principal_components  = 2, clusters = 2) 

feather.write_feather(groups,"groups.feather")
gNumber = 0
for group in groups:
    examples = random.choice(group,k=10)
    outFolder = pathlib.Path("output/" & gNumber).mkdir(parents=True, exist_ok=True)
    for file in examples:
        shutil.copy(file,outFolder)

    gNumber += 1