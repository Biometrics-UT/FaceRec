import time
from typing import List

from pathlib import *
import numpy as np
import face_recognition
import dlib

from src.utils import resize_image_exact, compute_encoding

assert dlib.DLIB_USE_CUDA


def compute_metric(original_enc, improved_enc):
    res = []
    for original, improved in zip(original_enc, improved_enc):
        if len(original) and len(improved):
            face_distances: np.ndarray = face_recognition.face_distance(original, improved[0])
            res.append(face_distances[0])
    from scipy import stats
    print(stats.describe(np.array(res)))


original_path = Path("../data/originals")
assert original_path.exists() and original_path.is_dir()
t = time.time()
original_files: List[Path] = [f for f in original_path.glob('**/*.jpg') if f.is_file()]
print("Loading filename took:", time.time() - t)
t = time.time()
original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
print("Resize original images took:", time.time() - t)
original_enc, loc1 = compute_encoding(original_images)

for i in range(1, 7):
    improved_path = Path("../data/improved" + str(i))
    print(improved_path.__str__())
    assert improved_path.exists() and improved_path.is_dir()
    improved_files: List[Path] = [f for f in improved_path.glob('**/*.png') if f.is_file()]
    print("There is", len(original_files), "originals and", len(improved_files), "improved")
    t = time.time()
    improved_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in improved_files]
    print("Resize improved images took:", time.time() - t)
    improved_enc, loc2 = compute_encoding(improved_images)
    compute_metric(original_enc, improved_enc)
