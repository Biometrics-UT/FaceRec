import time
from typing import List

from pathlib import *
import numpy as np
import face_recognition
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import resize_image_exact, compute_encoding
from shutil import copyfile

assert dlib.DLIB_USE_CUDA

N = 5658
original_path = Path("../data/originals")
assert original_path.exists() and original_path.is_dir()
t = time.time()
original_files: List[Path] = [f for f in original_path.glob('**/*.jpg') if f.is_file()][:N]
print("Loading filename took:", time.time() - t)
t = time.time()
original_images: List[np.ndarray] = [resize_image_exact(face_recognition.load_image_file(file.absolute().__str__()), 250, 250) for file in original_files]
image_mpl = [mpimg.imread(file) for file in original_files]
print("Resize original images took:", time.time() - t)
original_enc, loc1 = compute_encoding(original_images)
for i,el in enumerate(original_enc):
    if not el:
        print(i,original_files[i])
original_enc = [el[0] for el in original_enc]

distance_matrix = []
t = time.time()
to_exclude = set()
for i in range(len(original_files)):
    if i in to_exclude:
        continue
    face_distances: np.ndarray = face_recognition.face_distance(original_enc, original_enc[i])
    matching_faces = []
    for j, d in enumerate(face_distances):
        if d < 0.5:
            if j not in to_exclude:
                matching_faces.append(j)
            to_exclude.add(j)
    to_exclude.add(i)
    distance_matrix.append((i, matching_faces))

print("Done calculating distances in", time.time() - t)
save_path = Path("../data/classified")

for line in distance_matrix:
    folder_path = save_path / original_files[line[0]].stem
    folder_path.mkdir(parents=True, exist_ok=True)
    for img_index in line[1]:
        img_path = original_files[img_index]
        copyfile(img_path.absolute().__str__(), (folder_path / img_path.name).absolute().__str__())


def display_matrix(distance_matrix, image_mpl):
    nrows = len(distance_matrix)
    ncols = max([len(el[1]) for el in distance_matrix])
    print(nrows, ncols)
    figsize = [32, 32]
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    rows = fig.add_gridspec(nrows, 1)

    for x in range(nrows):
        current_row = distance_matrix[x][1]
        row = rows[x].subgridspec(1, len(current_row))
        for y, img_index in enumerate(current_row):
            img = image_mpl[img_index]
            ax = fig.add_subplot(row[0, y])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()
