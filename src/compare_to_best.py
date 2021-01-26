import json
import time
from typing import List

from pathlib import *
import numpy as np
import face_recognition
import dlib
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import resize_image_exact, compute_encoding, get_dict, get_best_encoding, remove_wrong_indexes

assert dlib.DLIB_USE_CUDA

best_encodings = get_best_encoding()


def get_score(original_path, type):
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
    res = {}
    for (identity, file_list) in d.items():
        other_encoding = [best_encodings[identity][2]]
        l = []
        for (file, index) in file_list:
            current_encoding = original_enc[index]
            face_distances: np.ndarray = face_recognition.face_distance(other_encoding, current_encoding)
            if face_distances != 0:  # remove the actual identity used
                l.append(face_distances[0])
        stat = stats.describe(l, axis=None)
        res[identity] = stat
    with open(f'stats_{original_path.name}.json', 'w') as outfile:
        json.dump(res, outfile, indent=2)
    return res


get_score(original_path=Path("../data/originals"), type="jpg")
for i in range(1, 7):
    get_score(original_path=Path("../data/improved" + str(i)), type="png")
