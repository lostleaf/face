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
from scipy.spatial.distance import cosine

def test():
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

def train():
    pca = PCA(n_components=0.95)
    faces_reg, reg_feats, face_reg_paths = dict(), dict(), dict()
    for path in glob("register_faces/*_crop/"):
        name = filter(None, path.split("/"))[-1][:-5]
        face_reg_paths[name] = glob(os.path.join(path + "*.jpg"))
        faces_reg[name] = [cv2.imread(img_path, 0) for img_path in face_reg_paths[name]] 
    data_train = [cv2.resize(face_img, (70, 70)).ravel() for faces in faces_reg.itervalues() for face_img in faces]
    pca.fit(data_train)
    for name, faces in faces_reg.iteritems():
        feats = [pca.transform(cv2.resize(face_img, (70, 70)).reshape((1, -1)))[0, :] for face_img in faces]
        reg_feats[name] = feats
    return pca, reg_feats, face_reg_paths

def save_det(img, img_path, face_pos):
    tmp = img.copy()
    for (x, y, w, h) in face_pos:
        cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("det_" + img_path, tmp)

def recognize(img_path, pca, face_det, reg_feats):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_pos = face_det.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    test_faces = [img_gray[y : y + h, x : x + w] for (x, y, w, h) in face_pos]

    test_feats = [pca.transform(cv2.resize(test_face, (70, 70)).reshape((1, -1)))[0, :] for test_face in test_faces]
    save_det(img, img_path, face_pos)
    dist_mat = []
    for idx, test_feat in enumerate(test_feats):
        median_dists = []
        for name, r_feats in reg_feats.iteritems():
            dists = []
            for feat in r_feats:
                dist = cosine(test_feat, feat)
                dists.append(dist)
            print "%s %d-th face, median dist to %s %f" % (img_path, idx, name, np.median(dists))
            median_dists.append(np.median(dists))
        dist_mat.append(median_dists)

    reg_names = reg_feats.keys()
    dist_mat = np.array(dist_mat)
    for line in np.argsort(dist_mat, axis=1):
        names = [reg_names[x] for x in line]
        print " ".join(names)

    for idx, test_face in enumerate(test_faces):
        plt.subplot(231 + idx)
        plt.imshow(test_face, cmap="gray")
    plt.show()

    recog = [None] * len(test_faces)
    used_name = set()
    used_face = set()
    for idx in np.argsort(dist_mat, axis=None):
        idx_x = idx / dist_mat.shape[1]
        idx_y = idx % dist_mat.shape[1]
        name = reg_names[idx_y]
        if name not in used_name and idx_x not in used_face:
            recog[idx_x] = names[idx_y] 
            used_name.add(name)
            used_face.add(idx_x)
    print recog

def main():
    pca, reg_feats, face_reg_paths = train()
    paths = glob("test/*.jpg")
    face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    for img_path in paths[:1]:
        recognize(img_path, pca, face_det, reg_feats)

if __name__ == "__main__":
    main()
