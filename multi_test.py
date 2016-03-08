import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from itertools import izip
from lbp import LBPHistFeat
from utils import show_face
from collections import Counter
from sklearn.decomposition import PCA

def train():
    lbp = LBPHistFeat()
    faces_reg, reg_feats, face_reg_paths = dict(), dict(), dict()
    for path in glob("register_faces/*_crop/"):
        name = filter(None, path.split("/"))[-1][:-5]
        face_reg_paths[name] = glob(os.path.join(path + "*.jpg"))
        faces_reg[name] = [cv2.imread(img_path, 0) for img_path in face_reg_paths[name]] 

    for name, faces in faces_reg.iteritems():
        feats = [lbp.transform(face_img) for face_img in faces]
        reg_feats[name] = feats
    return lbp, reg_feats, face_reg_paths

def save_det(img, img_path, face_pos):
    tmp = img.copy()
    for (x, y, w, h) in face_pos:
        cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("det_" + img_path, tmp)

def sort_pos(rows, cols):
    idx_up = np.where(rows < 100)[0]
    idx_down = np.where(rows >= 100)[0]
    ord_up = np.argsort(cols[idx_up])
    ord_down = np.argsort(cols[idx_down])
    return np.concatenate((idx_up[ord_up], idx_down[ord_down]), axis=0)

def recognize(img_path, lbp, face_det, reg_feats):
    ground_truth = img_path.split("(")[1].split(")")[0].split("_")
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_pos = face_det.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    ord_face = sort_pos(face_pos[:, 1], face_pos[:, 0])
    face_pos[:] = face_pos[ord_face, :]
    test_faces = [img_gray[y : y + h, x : x + w] for (x, y, w, h) in face_pos]

    test_feats = [lbp.transform(test_face) for test_face in test_faces]
    save_det(img, img_path, face_pos)
    dist_mat = []
    for idx, test_feat in enumerate(test_feats):
        min_dists = []
        for name, r_feats in reg_feats.iteritems():
            dists = []
            for feat in r_feats:
                dist = lbp.compare(test_feat, feat)
                dists.append(dist)
            # print "%s %d-th face, median dist to %s %f" % (img_path, idx, name, np.median(dists))
            min_dists.append(np.min(dists))
        dist_mat.append(min_dists)

    reg_names = reg_feats.keys()
    dist_mat = np.array(dist_mat)
    print reg_names
    print dist_mat

    recog = [None] * len(test_faces)
    used_name = set()
    used_face = set()
    for idx in np.argsort(dist_mat, axis=None):
        idx_x = idx / dist_mat.shape[1]
        idx_y = idx % dist_mat.shape[1]
        name = reg_names[idx_y]
        if name not in used_name and idx_x not in used_face and dist_mat[idx_x, idx_y] < 0.08:
            recog[idx_x] = reg_names[idx_y] 
            used_name.add(name)
            used_face.add(idx_x)
    correct = recog[:3] == ground_truth and all((x is None for x in recog[3:]))
    print correct
    print recog

    for idx, test_face in enumerate(test_faces):
        plt.subplot(231 + idx)
        plt.imshow(test_face, cmap="gray")
    plt.show()

    for idx, ((x, y, w, h), recog_name) in enumerate(izip(face_pos, recog)):
        if recog_name is not None:
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, recog_name, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    return correct

def main():
    lbp, reg_feats, face_reg_paths = train()
    paths = glob("test/*.jpg")
    face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    cnt = 0
    for img_path in paths:
        ret = recognize(img_path, lbp, face_det, reg_feats)
        if ret:
            cnt += 1
    print cnt

if __name__ == "__main__":
    main()
