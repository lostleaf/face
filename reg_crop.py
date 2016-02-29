import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

class FaceCropper:
    def __init__(self):
        self.face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

    #for debug
    def show_face(img, face_pos):
        for (x, y, w, h) in face_pos:
            plt.imshow(img[y : y + h, x : x + w, ::-1])
            plt.show()
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        plt.imshow(img[:, :, ::-1])
        plt.show()

    def crop_faces(self, img_paths):
        faces = (self.crop_face(img_path) for img_path in img_paths)
        return [face for face in faces if face is not None]

    def crop_face(self, img_path):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        factor=1.1
        while 1:
            face_pos = self.face_det.detectMultiScale(img_gray, scaleFactor=factor, minNeighbors=3, minSize=(30, 30))
            if len(face_pos) > 1:
                factor += .1
            else:
                break

        # self.show_face(img, face_pos)

        if len(face_pos):
            (x, y, w, h) = face_pos[0]
            img_face = img[y : y + h, x : x + w, :]
            return img_face
        return None

def main():
    dir_names = os.listdir("register_faces")
    cropper = FaceCropper()
    for dir_name in dir_names:
        dir_path = "register_faces/" + dir_name
        # print dir_path, os.path.isdir(dir_path)
        if os.path.isdir(dir_path) and not dir_name.endswith("_crop") and dir_name + "_crop" not in dir_names:
            print "Processing " + dir_name
            os.mkdir(dir_path + "_crop")
            img_paths = glob(dir_path + "/*.jpg")
            img_faces = cropper.crop_faces(img_paths)
            print "%d faces collected" % len(img_faces)
            for idx, img_face in enumerate(img_faces):
                cv2.imwrite(dir_path + "_crop/%d.jpg" % idx, img_face)

if __name__ == "__main__":
    main()
