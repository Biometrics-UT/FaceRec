import json
import time
from pathlib import *
from typing import List

import dlib
import face_recognition
import numpy as np
from scipy import stats

from src.utils import resize_image_exact, compute_encoding, get_dict

assert dlib.DLIB_USE_CUDA

import os

os.makedirs("stats", exist_ok=True)


def get_best():
    original_path = Path("../data/originals")
    type = "jpg"
    assert original_path.exists() and original_path.is_dir()
    t = time.time()
    original_files: List[Path] = [f for f in original_path.glob('**/*.' + type) if f.is_file()]
    print("Loading filename took:", time.time() - t)
    t = time.time()
    original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
    print("Resize original images took:", time.time() - t)
    original_enc, loc1 = compute_encoding(original_images)
    for i, el in enumerate(original_enc):
        if not el:
            print(i, original_files[i])
    original_enc = [el[0] for el in original_enc]

    d = get_dict(original_path, type)
    best_representation = {}
    for k, v in d.items():
        other_enc = [original_enc[i] for (file, i) in v]
        l = []
        for i, el in enumerate(other_enc):
            face_distances: np.ndarray = face_recognition.face_distance(other_enc, el)
            l.append((str(v[i][0]), stats.describe(face_distances)))
        l.sort(key=lambda x: x[1].mean)
        best_representation[k] = l[0][0]
    with open('stats/best.json', 'w') as outfile:
        json.dump(best_representation, outfile, indent=2)


if __name__ == '__main__':
    get_best()
