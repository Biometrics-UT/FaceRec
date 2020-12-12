import cv2
from pathlib import *
import numpy as np

directory_path = Path("../data/fgrc_dataset/Spring2003")
cascade_path = Path("../models/default_frontalface.xml")

assert cascade_path.exists() and cascade_path.is_file()
assert directory_path.exists() and directory_path.is_dir()


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


print(cascade_path.absolute().__str__())
faceCascade = cv2.CascadeClassifier(cascade_path.absolute().__str__())

list_files = [f for f in directory_path.glob('**/*.jpg') if f.is_file()]

file = np.random.choice(list_files)
print(f"Using {file} as the file")

image = cv2.imread(file.absolute().__str__())
image = resize_image(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Faces found", image)

cv2.waitKey(0)
