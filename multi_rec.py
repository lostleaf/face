import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from itertools import izip
from lbp import LBPHistFeat
from utils import show_face
from collections import Counter

def main():
    feat_extr = LBPHistFeat()

    img_test = cv2.imread("test1.jpg")
    img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    face_pos = face_det.detectMultiScale(img_test_gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    faces_test_gray = [img_test_gray[y : y + h, x : x + w] for (x, y, w, h) in face_pos]
    test_feats = [feat_extr.extract(cv2.resize(img, (100, 100))) for img in faces_test_gray]
    # show_face(img_test, face_pos)

    faces_reg, reg_feats, face_reg_paths = dict(), dict(), dict()
    for path in glob("register_faces/*_crop/"):
        name = filter(None, path.split("/"))[-1][:-5]
        face_reg_paths[name] = glob(os.path.join(path + "*.jpg"))
        faces_reg[name] = [cv2.imread(img_path, 0) for img_path in face_reg_paths[name]] 
        reg_feats[name] = [feat_extr.extract(cv2.resize(img, (100, 100))) for img in faces_reg[name]]
    
    reg_det_median = []
    names = reg_feats.keys()
    for name, r_feats in reg_feats.iteritems():
        reg_det_dists = []
        print name
        for (reg_feat, reg_path) in izip(r_feats, face_reg_paths[name]):
            dists = [feat_extr.compare(reg_feat, test_feat) for test_feat in test_feats]
            if dists[2] > dists[1] or dists[2] > dists[0]:
                print dists, reg_path
            reg_det_dists.append(dists)
            # vote_idx = np.argmin(dists)
            # vote_dist = dists[vote_idx]
            # if vote_dist < 0.18:
            #     votes[vote_idx].append((name, vote_dist, reg_path))
            # votes[vote_idx].append((name, vote_dist))
        reg_det_median.append(np.median(reg_det_dists, axis=0))
    reg_det_median = np.asarray(reg_det_median)
    print reg_det_median

    print names
    used = set()
    recog = [None] * len(test_feats)
    for idx in np.argsort(reg_det_median, axis=None):
        idx_x = idx / reg_det_median.shape[1]
        idx_y = idx % reg_det_median.shape[1]
        name = names[idx_x]
        if name not in used:
            recog[idx_y] = names[idx_x] 
            used.add(name)
    print recog

    for idx, ((x, y, w, h), recog_name) in enumerate(izip(face_pos, recog)):
        if recog_name is not None:
            color = (0, 255, 0)
            cv2.rectangle(img_test, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_test, recog_name, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
        else:
            cv2.rectangle(img_test, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.imshow(img_test[:, :, ::-1])
    plt.show()
    cv2.imwrite("test1_rec.jpg", img_test)

if __name__ == "__main__":
    main()
