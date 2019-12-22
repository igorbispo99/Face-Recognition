import numpy as np
import cv2 as cv
import argparse
import pickle
import face_detector as fd

from joblib import load
from keras.models import load_model

DL_CLASS = load_model("me_detector.h5")
WIN_NAME = "Face Classifier"

def classify_faces(img):
    faces, _ = fd.detect_faces_cascade(np.copy(img))

    if len(faces) == 0:
        return img
    
    crop_face = fd.crop_face(img, faces)
    crop_face = 1/255 * cv.cvtColor(cv.resize(crop_face, (160, 160)), cv.COLOR_BGR2RGB)
    

    is_my_face = DL_CLASS.predict(np.expand_dims(crop_face, 0))[0]
    
    print(is_my_face)

    result = np.argmax(is_my_face)

    if result == 0 and is_my_face[result] > 0.98:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    
    (x,y,w,h) = faces[0]

    cv.rectangle(img, (x, y), (x+w, y+h), color, 2)  

    return img

def main():
    webcm = cv.VideoCapture(1)

    webcm.grab()
    webcm.retrieve()

    cv.namedWindow(WIN_NAME)

    while True:
        key = cv.waitKey(1)

        if key == ord('q'):
            break

        webcm.grab()
        ret, frame = webcm.retrieve()
        img_cascade = classify_faces(frame)

        cv.imshow(WIN_NAME, img_cascade)
    
if __name__ == '__main__':
    main()
