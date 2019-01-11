import numpy as np
import cv2 as cv
import argparse
import pickle
import face_detector as fd

from joblib import load

ADA_CLASS = load("adaboost_face.ada")
RF_CLASS = load("randomforest_face.rf")
NB_CLASS = load("naivebayes_face.nb")
MLP_CLASS = load("mlp_face.mlp")
WIN_NAME = "Face Classifier"

def classify_faces(img):
    faces, _ = fd.detect_faces_cascade(np.copy(img))

    if len(faces) == 0:
        return img
    
    crop_face = fd.crop_face(img, faces)
    crop_face = cv.resize(crop_face, (110, 110)).ravel()

    is_my_face = NB_CLASS.predict(crop_face.reshape(1, crop_face.shape[0]))

    if is_my_face == 1:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    
    (x,y,w,h) = faces[0]

    cv.rectangle(img, (x, y), (x+w, y+h), color, 2)  

    return img

def main():
    webcm = cv.VideoCapture(0)

    cv.namedWindow(WIN_NAME)

    while True:
        key = cv.waitKey(1)

        if key == ord('q'):
            break

        ret, frame = webcm.read()
        img_cascade = classify_faces(frame)

        cv.imshow(WIN_NAME, img_cascade)
    
if __name__ == '__main__':
    main()