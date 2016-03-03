import cv2
import matplotlib.pyplot as plt

def show_face(img, face_pos):
    img = img.copy()
    for (x, y, w, h) in face_pos:
        plt.imshow(img[y : y + h, x : x + w, ::-1])
        plt.show()
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    plt.imshow(img[:, :, ::-1])
    plt.show()
