import time
from typing import List

import cv2
from pathlib import *
import numpy as np
import face_recognition
import dlib

assert dlib.DLIB_USE_CUDA
out_path = Path("../data/out")
assert out_path.exists() and out_path.is_dir()
in_path = Path("../data/in")
assert in_path.exists() and in_path.is_dir()

t = time.time()
out_files: List[Path] = [f for f in out_path.glob('**/*.jpg') if f.is_file()]
in_files: List[Path] = [f for f in in_path.glob('**/*.png') if f.is_file()]
print("Loading filename took:", time.time() - t)


def resize_image(_image, width=None, height=None, inter=cv2.INTER_AREA):
    (_h, _w) = _image.shape[:2]

    if width is None and height is None:
        return _image
    if width is None:
        r = height / float(_h)
        dim = (int(_w * r), height)
    else:
        r = width / float(_w)
        dim = (width, int(_h * r))

    return cv2.resize(_image, dim, interpolation=inter)


def resize_image_exact(_image, width, height, inter=cv2.INTER_AREA):
    resized = cv2.resize(_image, (width, height), interpolation=inter)
    return resized[:, :, ::-1]  # invert back colors


t = time.time()
out_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in out_files]
print("Resize image out took:", time.time() - t)
t = time.time()
in_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in in_files]
print("Resize image in took:", time.time() - t)


def compute_encoding(images):
    _t = time.time()
    face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample=0, batch_size=128)
    face_encodings = []
    for _image, _location in zip(images, face_locations):
        face_encodings.append(face_recognition.face_encodings(_image, _location))
    print("Compute face encoding took:", time.time() - _t)
    return face_encodings, face_locations


enc1, loc1 = compute_encoding(out_images)
enc2, loc2 = compute_encoding(in_images)
res = []
for e1, e2 in zip(enc1, enc2):
    if len(e1) and len(e2):
        face_distances: np.ndarray = face_recognition.face_distance(e1, e2[0])
        res.append(face_distances[0])

from scipy import stats

print(stats.describe(np.array(res)))
