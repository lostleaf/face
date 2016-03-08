import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from glob import glob
from itertools import izip

class LBPHistFeat:
    def __init__(self, radius=2, method="default"):
        self.radius = radius
        self.n_points = 8 * radius
        if method == "uniform":
            self.n_bins = self.n_points + 2
        if method == "default":
            self.n_bins = 256 ** radius
        self.method = method

    def extract(self, img_gray):
        lbp = local_binary_pattern(img_gray, self.n_points, self.radius, self.method)
        # print lbp.max()
        hist, _ = np.histogram(lbp, normed=True, bins=self.n_bins, range=(0, self.n_bins))
        return hist

    def transform(self, img_gray):
        return self.extract(img_gray)
    
    def kldivergence(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    
    def chi_square(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum((p[filt] - q[filt]) ** 2 / (p[filt] + q[filt]))

    def compare(self, hist1, hist2):
        return self.chi_square(hist1, hist2)

def main():
    feat = LBPHistFeat()

    img_test = cv2.imread("test.jpg")
    img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    face_pos = face_det.detectMultiScale(img_test_gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    faces_test_gray = [img_test_gray[y : y + h, x : x + w] for (x, y, w, h) in face_pos]
    test_feats = [feat.extract(img) for img in faces_test_gray]

    face_ql_imgs = [cv2.imread(img_path, 0) for img_path in glob("register_faces/Qinglin_crop/*.jpg")]
    ql_feats = [feat.extract(img) for img in face_ql_imgs]

    sim_mat = np.array([[feat.compare(ql_feat, test_feat) for ql_feat in ql_feats] for test_feat in test_feats])
    print sim_mat
    var = np.var(sim_mat, axis=0)
    # print var
    # print np.count_nonzero(var < 2e-4), np.where(var < 2e-4)
    print np.median(sim_mat, axis=1)
    ql_idx = np.argmin(np.median(sim_mat, axis=1))
    for idx, (x, y, w, h) in enumerate(face_pos):
        col = (0, 255, 0) if idx == ql_idx else (0, 0, 255)
        cv2.rectangle(img_test, (x, y), (x + w, y + h), col, 2)
    plt.imshow(img_test[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    main()
