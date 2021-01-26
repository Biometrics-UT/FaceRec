import time
from typing import List

from pathlib import *
import numpy as np
import face_recognition
import dlib
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import resize_image_exact, compute_encoding, get_dict

assert dlib.DLIB_USE_CUDA


def get_score(original_path, type,orign=None):
    print(original_path.__str__())
    assert original_path.exists() and original_path.is_dir()
    t = time.time()
    original_files: List[Path] = [f for f in original_path.glob('**/*.' + type) if f.is_file()][:100]
    print("Loading filename took:", time.time() - t)
    t = time.time()
    original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
    print("Resize original images took:", time.time() - t)
    original_enc, loc1 = compute_encoding(original_images)
    original_enc = [el[0] for el in original_enc]

    d = get_dict(original_path, type)

    for k, v in d.items():
        other_enc = [original_enc[i] for (file, i) in v]
        if orign!=None:
            origin_enc=[orign[i] for (file, i) in v]
        l = []
        for el in other_enc:
            if orign!=None:
                face_distances: np.ndarray = face_recognition.face_distance(origin_enc, el)
            else:
                face_distances: np.ndarray = face_recognition.face_distance(other_enc, el)
            l.append(face_distances)
        print(stats.describe(l, axis=None))
        break
    return original_enc


original_enc=get_score(original_path=Path("../data/originals"), type="jpg")
for i in range(1, 7):
    get_score(original_path=Path("../data/improved" + str(i)), type="png",orign=original_enc)
