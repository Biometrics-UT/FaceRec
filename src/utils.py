import json
import time

import cv2
import dlib
import face_recognition
import numpy as np

from collections import defaultdict
from typing import List
from pathlib import Path

assert dlib.DLIB_USE_CUDA


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


def compute_encoding(images):
    _t = time.time()
    face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample=0, batch_size=128)
    face_encodings = []
    for _image, _location in zip(images, face_locations):
        face_encodings.append(face_recognition.face_encodings(_image, _location))
    print("Compute face encoding took:", time.time() - _t)
    return face_encodings, face_locations


def get_dict(path, typ="jpg", ignore: List[Path] = None):
    """
    Get the identity with all the associated images
    :param ignore: a list of path to ignore
    :param path: the path to the folder to have the identities in
    :param typ: the type of image to consider
    :return: a dictionary with as key the identity and as value a list of pair with the file path and the global index of the file
    """
    if ignore is None:
        ignore = []
    assert path.exists() and path.is_dir()
    original_files: List[Path] = [f for f in path.glob('**/*.' + typ) if f.is_file() if f not in ignore]

    d = defaultdict(list)
    for i, file in enumerate(original_files):
        d[file.name.split("d")[0]].append((file, i))
    return d


def get_best_encoding():
    """
    Calculate the best encoding based on a json file
    :return: a dictionary with as key the current identity and as value a triplet as the file uri, the file path and the encoding
    """
    with open("stats/best.json") as in_file:
        js: dict = json.load(in_file)
    original_files: List[Path] = [Path(f) for f in js.values()]
    original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
    original_enc, loc1 = compute_encoding(original_images)
    res = {}
    for i, (k, enc) in enumerate(zip(js.keys(), original_enc)):
        if not enc:
            print(k, "failed with", enc)
        res[k] = (js[k], original_files[i], enc[0])
    return res


def remove_wrong_indexes(to_check: list, *args: list) -> List[tuple]:
    """
    Remove the wrong id from to_check and all list pass as varargs
    :param to_check: a list with potential harmful element
    :param args: any list that have the same lenght as tocheck
    :return: a list of tuple containing the index of the element removed,
     the element removed from to check and then in order the elements form args
    """
    for l in args:
        if len(l)!=len(to_check):
            raise Exception("You shouldn't give different size of list")
    to_remove_index = []
    for i, el in enumerate(to_check):
        if not el:
            to_remove_index.append(i)
    res: List[tuple] = []
    for offset, index in enumerate(to_remove_index):
        cur_rem: tuple = (index,)
        cur_rem += (to_check.pop(index - offset),)
        for l in args:
            cur_rem += (l.pop(index - offset),)
        res.append(cur_rem)
    return res
