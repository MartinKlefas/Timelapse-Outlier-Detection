from concurrent.futures import ThreadPoolExecutor
import pathlib,shutil,uuid

def do_one_move(file : str, destFolder : pathlib.Path):
    try:
        fileObj = pathlib.Path(file)
        fileSuffix = fileObj.suffix
        shutil.copy(file,str(destFolder / str(uuid.uuid4()) + fileSuffix))
    except:
        ...

def move_batch(sourceBatch, destFolder):
    
    with ThreadPoolExecutor() as executor:
        executor.map(do_one_move, sourceBatch, destFolder)