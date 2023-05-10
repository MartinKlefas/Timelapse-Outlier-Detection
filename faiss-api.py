from fastapi import File
from fastapi import UploadFile
from fastapi import Form
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

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

import uvicorn, random, datetime

if __name__ == '__main__':
    print("Starting server")
    uvicorn.run("faiss-api:app", host="0.0.0.0", port=8080, timeout_keep_alive=120)
else:
    start_time = datetime.now()
    try:  
        res = faiss.StandardGpuResources()  
        gpu_index_flat = faiss.read_index("faiss-test/index.index")
        print(f"Index loaded, {gpu_index_flat.ntotal} vectors present.")
    except:
        print(f"Failed to load index. Exiting")
        sys.exit()

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


    format_num_images = format(gpu_index_flat.ntotal,",")

    msg += f"{minutes:02d}:{seconds:02d}, and is searcing on {format_num_images} images."

  
    return {"message": msg}