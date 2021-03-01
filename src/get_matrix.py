import itertools
import time
from pathlib import *
from typing import List

import dlib
import face_recognition
import numpy as np

from src.utils import resize_image_exact, compute_encoding, get_dict, remove_wrong_indexes

assert dlib.DLIB_USE_CUDA

import os

os.makedirs("data_metrics", exist_ok=True)

def get_matrix(original_path, type):
    print(original_path.__str__())
    assert original_path.exists() and original_path.is_dir()
    t = time.time()
    original_files: List[Path] = [f for f in original_path.glob('**/*.' + type) if f.is_file()]
    print("Loading filename took:", time.time() - t)
    t = time.time()
    original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
    print("Resize original images took:", time.time() - t)
    original_enc, loc1 = compute_encoding(original_images)
    wrong_stuff = remove_wrong_indexes(original_enc, original_images, original_files)
    print("removed", [el[0] for el in wrong_stuff])
    original_enc = [el[0] for el in original_enc]

    d = get_dict(original_path, type, [el[3] for el in wrong_stuff])
    matrix = []
    for (identity, file_list) in d.items():
        for (file, index) in file_list:
            current_encoding = original_enc[index]
            face_distances: np.ndarray = face_recognition.face_distance(original_enc, current_encoding)
            matrix.append([str(el) for el in face_distances])
    with open(f'data_metrics/matrix_{original_path.name}.txt', 'w') as outfile:
        for l in matrix:
            outfile.write(" ".join(l) + "\n")
    with open(f'data_metrics/id_{original_path.name}.txt', 'w') as outfile:
        outfile.write(" ".join(list(itertools.chain.from_iterable(itertools.repeat(k, len(v)) for (k, v) in d.items()))) + "\n")

    return matrix


def get_matrices():
    get_matrix(original_path=Path("../data/originals"), type="jpg")
    for i in range(1, 7):
        get_matrix(original_path=Path("../data/improved" + str(i)), type="png")


if __name__ == '__main__':
    get_matrices()
