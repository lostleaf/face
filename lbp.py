import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from glob import glob
from itertools import izip

RADIUS = 3
N_POINTS = 8 * RADIUS
N_BINS = N_POINTS + 2

def kldivergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def get_test_faces(img_test):
    img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    face_pos = face_det.detectMultiScale(img_test_gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    # for (x, y, w, h) in face_pos:
    #     plt.imshow(img_test[y : y + h, x : x + w, ::-1])
    #     plt.show()
    #     cv2.rectangle(img_test,(x,y),(x+w,y+h),(255,0,0),2)
    # plt.imshow(img_test[:, :, ::-1])
    # plt.show()
    faces_gray = [img_test_gray[y : y + h, x : x + w] for (x, y, w, h) in face_pos]
    return faces_gray, face_pos

def get_lbp_hists(imgs):
    hists = []
    for img_gray in imgs:
        lbp = local_binary_pattern(img_gray, N_POINTS, RADIUS, "uniform")
        hist, _ = np.histogram(lbp, normed=True, bins=N_BINS, range=(0, N_BINS))
        hists.append(hist)
        # plt.imshow(lbp, cmap="gray")
        # plt.show()
    return hists

def get_hists_ql():
    img_paths = glob("register_faces/Qinglin_crop/*.jpg") 
    imgs_gray = [cv2.imread(img_path, 0) for img_path in img_paths]
    return get_lbp_hists(imgs_gray)

def main():
    img_test = cv2.imread("test.jpg")
    faces_test_gray, face_pos = get_test_faces(img_test)
    face_test_hists = get_lbp_hists(faces_test_gray)
    face_ql_hists = get_hists_ql()
    sim_mat = np.array([[kldivergence(hist_test, hist_ql) for hist_ql in face_ql_hists] for hist_test in face_test_hists])
    ql_idx = np.argmin(np.median(sim_mat, axis=1))
    for idx, (x, y, w, h) in enumerate(face_pos):
        col = (0, 255, 0) if idx == ql_idx else (0, 0, 255)
        cv2.rectangle(img_test, (x, y), (x + w, y + h), col, 2)
    plt.imshow(img_test[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    main()
