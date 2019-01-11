import cv2 as cv
import numpy as np
import pickle
import argparse
import face_detector as fd
import glob

def init_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("-n", required=True, 
                          help="Number of images to extract from folder.",
                          type=int)
    
    ap.add_argument("-f", "--folder",
                        required = True,
                        help="Folder to extract images.")
    
    return ap, ap.parse_args()

def extract_face(img):
    faces, _ = fd.detect_faces_cascade(np.copy(img))

    if len(faces) == 0:
        return None

    cropped_face = fd.crop_face(img, faces) 
    
    return cropped_face

def select_faces(folder, n):
    img_list = glob.glob(folder + "/*.jpg")

    faces_list = []

    for i in range(n):
        img_to_extract = np.random.choice(img_list)
        img = cv.imread(img_to_extract)
        
        face = extract_face(img)

        while face is None:
            img_to_extract = np.random.choice(img_list)
            img = cv.imread(img_to_extract)
            
            face = extract_face(img)
        
        faces_list.append(face)

    return faces_list

def main():
    ap, args = init_args()

    faces_list = select_faces(args.folder, args.n)

    file_name = str(args.n) + "_faces"

    with open(file_name, 'wb') as out:
        pickle.dump(faces_list, out)

if __name__ == '__main__':
    main()
