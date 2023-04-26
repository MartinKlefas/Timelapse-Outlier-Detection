from concurrent.futures import ThreadPoolExecutor
import pathlib
from PIL import Image

size = 224,224

def resize(image : pathlib.Path):
        global size
        thumbname = str(image).replace("GoPro TimeLapse","thumbnails")
        if not pathlib.Path(thumbname).exists():
            try:
                im = Image.open(image)
                thumbfolder = str(image.parents[0]).replace("GoPro TimeLapse","thumbnails")
                
                
                x=pathlib.Path(thumbfolder).mkdir(parents=True, exist_ok=True)
                im.thumbnail(size,Image.Resampling.LANCZOS)
                im.save(thumbname)
            except:
                  ...
        
        return str(image)

files = pathlib.Path("Y:/Allotment Timelapse/GoPro TimeLapse").rglob("*.jpg")

with ThreadPoolExecutor() as executor:
        features_list = list(executor.map(resize, files))