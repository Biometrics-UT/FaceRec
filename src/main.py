import cv2
from pathlib import *
import numpy as np
import face_recognition

directory_path = Path("../data/nd1/Spring2003")
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




list_files = [f for f in directory_path.glob('**/*.jpg') if f.is_file()]

file = np.random.choice(list_files)
print(f"Using {file} as the file")
## OPENCV
file_path_str = file.absolute().__str__()


def opencv(file_path_str):
    cascade_path = Path("../models/default_frontalface.xml")
    assert cascade_path.exists() and cascade_path.is_file()
    faceCascade = cv2.CascadeClassifier(cascade_path.absolute().__str__())
    image = cv2.imread(file_path_str)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Found {len(faces)} faces!")
    print(*faces)
    return image,faces


def draw_faces(_image, _faces):
    # Draw a rectangle around the faces
    for (x, y, w, h) in _faces:
        cv2.rectangle(_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    _image = resize_image(_image, height=600)
    cv2.imshow("Faces found", _image)
    cv2.waitKey(0)


# draw_faces(image,faces)

## FACE RECOGNITION
def draw_faces_scaled(face_locations, face_names, face_landmarks_list, frame, scale=4):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for (top, right, bottom, left), name, face_landmarks in zip(face_locations, face_names, face_landmarks_list):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Draw some eyebrow
        cv2.polylines(frame, np.int32([face_landmarks['left_eyebrow']]) * scale, isClosed=True, color=(68, 54, 39, 128))
        cv2.polylines(frame, np.int32([face_landmarks['right_eyebrow']]) * scale, isClosed=True,
                      color=(68, 54, 39, 128))
        cv2.polylines(frame, np.int32([face_landmarks['left_eyebrow']]) * scale, isClosed=True, color=(68, 54, 39, 150),
                      thickness=5)
        cv2.polylines(frame, np.int32([face_landmarks['right_eyebrow']]) * scale, isClosed=True,
                      color=(68, 54, 39, 150), thickness=5)

        # Gloss the lips
        cv2.polylines(frame, np.int32([face_landmarks['top_lip']]) * scale, isClosed=True, color=(150, 0, 0, 128))
        cv2.polylines(frame, np.int32([face_landmarks['bottom_lip']]) * scale, isClosed=True, color=(150, 0, 0, 128))
        cv2.polylines(frame, np.int32([face_landmarks['top_lip']]) * scale, isClosed=True, color=(150, 0, 0, 64),
                      thickness=5)
        cv2.polylines(frame, np.int32([face_landmarks['bottom_lip']]) * scale, isClosed=True, color=(150, 0, 0, 64),
                      thickness=5)

        # Sparkle the eyes
        cv2.polylines(frame, np.int32([face_landmarks['left_eye']]) * scale, isClosed=True, color=(255, 255, 255, 30))
        cv2.polylines(frame, np.int32([face_landmarks['right_eye']]) * scale, isClosed=True, color=(255, 255, 255, 30))

    _image = resize_image(frame, height=600)
    cv2.imshow("Faces found", _image)
    cv2.waitKey(0)


image = face_recognition.load_image_file(file_path_str)
image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
image_small = image_small[:, :, ::-1]  # BGR to RGB (opencv fix)
known_face_encodings = []
known_face_names = []
face_locations = face_recognition.face_locations(image_small)
face_landmarks_list = face_recognition.face_landmarks(image_small)
face_encodings = face_recognition.face_encodings(image_small, face_locations)
face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    if len(face_distances):
        best_match_index: int = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    face_names.append(name)

print(face_locations)
print(face_landmarks_list)

draw_faces_scaled(face_locations, face_names, face_landmarks_list, image)
