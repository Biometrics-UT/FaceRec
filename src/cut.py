import time
from math import floor

import cv2
from pathlib import *
import numpy as np
import face_recognition
from PIL import Image

directory_path = Path("../data/nd1/Spring2003")
out_path = Path("../data/out")
out_path.mkdir(exist_ok=True)

assert directory_path.exists() and directory_path.is_dir()
assert out_path.exists() and out_path.is_dir()

list_files = [f for f in directory_path.glob('**/*.jpg') if f.is_file()]
# file: Path = np.random.choice(list_files)
t = time.time()
nbr = len(list_files)
for i, file in enumerate(list_files):
    # print(f"Using {file} as the file")
    file_path_str = file.absolute().__str__()
    image = face_recognition.load_image_file(file_path_str)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="hog")  # cnn if gpu
    # print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for j, face_location in enumerate(face_locations):
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[max(top-200,0):min(bottom + 50, image.shape[0]), max(left - 100,0):min(right + 100, image.shape[1])]
        pil_image = Image.fromarray(face_image)
        # print(out_path / f"{file.stem}_{i}{file.suffix}")
        pil_image.save(out_path / f"{file.stem}_{j}{file.suffix}")
    if i % (nbr // 100) == 0:
        print(f"Done at {i / nbr * 100:.2f}% in {time.time() - t:.2f}s")
print(f"Done at 100% in {time.time() - t:.2f}s")
