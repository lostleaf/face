import cv2
import os
import numpy as np
from lbp import LBPHistFeat
from glob import glob
from itertools import izip
from operator import itemgetter
from collections import Counter

REG_NUM = 5
class NN:
    def __init__(self, feat, n_neighbor=REG_NUM):
        self.feat_extr = feat
        self.n_neighbor = n_neighbor

    def fit(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def predict(self, test_feats):
        rets = []
        for test_feat in test_feats:
            dists = [(self.feat_extr.compare(test_feat, feat), label) for feat, label in izip(self.feats, self.labels)]
            # print dists
            dists.sort(key=itemgetter(0))
            top_labels = [label for _, label in dists[:self.n_neighbor]]
            label_pred = Counter(top_labels).most_common(1)[0][0]
            rets.append(label_pred)
        return rets

def main():
    feat_extr = LBPHistFeat(2, "default")
    names = [name for name in os.listdir("att_faces") if os.path.isdir("att_faces/" + name)]
    faces = {n : [cv2.imread(p, 0) for p in glob("att_faces/%s/*.pgm" % n)] for n in names}
    face_feats = {n : [feat_extr.extract(img) for img in v] for n, v in faces.iteritems()}

    feats_reg, labels_reg, feats_test, labels_test = [], [], [], []
    for name, feats in face_feats.iteritems():
        for feat in feats[:REG_NUM]:
            feats_reg.append(feat)
            labels_reg.append(name)
        for feat in feats[REG_NUM:]:
            feats_test.append(feat)
            labels_test.append(name)
    
    nn = NN(feat_extr, 1)
    nn.fit(feats_reg, labels_reg)
    labels_pred = nn.predict(feats_test)
    print labels_test
    print labels_pred

    cnt = 0
    for label_pred, label_test in izip(labels_pred, labels_test):
        if label_pred == label_test:
            cnt += 1

    print "register number %d, accuracy %f" % (REG_NUM, float(cnt) / len(labels_pred))

if __name__ == "__main__":
    main()
