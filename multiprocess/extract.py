import numpy as np
from keras.models import Model
import pathlib, pickle


def features_extract(imageData: np.ndarray, model_ft : Model,new_pickles_folder : pathlib.Path, theseFileNames, i):
    if imageData.shape[0] > 0:
        batch_features = model_ft.predict(imageData, use_multiprocessing=True, verbose=1)
    
    with open(str(new_pickles_folder/ f"features_batch_{i}.pickle"), 'wb') as handle:
                    pickle.dump(batch_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(new_pickles_folder/ f"filenames_batch_{i}.pickle"), 'wb') as handle:
                    pickle.dump(theseFileNames, handle, protocol=pickle.HIGHEST_PROTOCOL)