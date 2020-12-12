import cv2
from pathlib import *
import numpy as np

directory_path = Path("../data/fgrc_dataset/Spring2003")
cascade_path = Path("../models/default_frontalface.xml")

assert cascade_path.exists() and cascade_path.is_file()
assert directory_path.exists() and directory_path.is_dir()

faceCascade = cv2.CascadeClassifier(cascade_path)

list_files=[f for f in directory_path.glob('**/*.jpg') if f.is_file()]

file=np.random.choice(list_files)
print(f"Using {file} as the file")

image = cv2.imread(file)
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
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)