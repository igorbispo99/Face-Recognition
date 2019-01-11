from pathlib import Path

import numpy as np
import cv2 as cv
import argparse
import pickle

INTERVAL = 1
OUT_FILE = 'dumped_imgs'
WIN_WEB = 'Webcam Output'
WIN_FACE = 'Face Output'
F_CASCADE = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def dump_file_exists():
    f = Path(OUT_FILE)
    return f.is_file()

def load_dumped_imgs():
    imgs = []

    if dump_file_exists():
        _img  = open(OUT_FILE, 'rb')
        imgs = pickle.load(_img)
    else:
        _img  = open(OUT_FILE, 'wb')
        pickle.dump(imgs, _img)

    _img.close()

    return imgs

def save_dumped_imgs(imgs):
    with open(OUT_FILE, 'wb') as _img:
        pickle.dump(imgs, _img)

def detect_faces_cascade(img, color = (0, 0, 255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = F_CASCADE.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)   

    return faces, img

def crop_face(img, faces):
    # Assuming that exists only one face in the image

    (x, y, w, h) = faces[0]

    croped_img = img[y:y+w, x:x+h]

    return croped_img

def take_pics(obj_cam, s_imgs, n = 1):
    i = 0

    while i < n:
        _ = cv.waitKey(int(INTERVAL * 1000))

        ret, frame = obj_cam.read()

        faces, new_frame = detect_faces_cascade(np.copy(frame))

        if len(faces) == 0:
            continue

        cropped_img = crop_face(frame, faces)
        s_imgs.append(cropped_img)

        cv.imshow(WIN_FACE, cropped_img)

        i += 1

def prepare_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", 
                    required=False,
                    help="Interval between captures (in seconds)",
                    type=int)
    
    return ap, ap.parse_args()

def main():
    ap, args = prepare_args()

    if args.interval:
        global INTERVAL
        INTERVAL = args.interval

    imgs = load_dumped_imgs()

    webcm = cv.VideoCapture(0)

    cv.namedWindow(WIN_WEB)

    while True:
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('t'):
            take_pics(webcm, imgs, 5)
        else:
            ret, frame = webcm.read()

            img_cascade = np.copy(frame)
            _, img_cascade = detect_faces_cascade(img_cascade)

            cv.imshow(WIN_WEB, img_cascade)
    
    save_dumped_imgs(imgs)
    
if __name__ == '__main__':
    main()
        