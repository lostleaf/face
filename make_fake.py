import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import shuffle

def concat_imgs(img1, img2, img3):
    padding = np.zeros((img1.shape[0], 15, 3), dtype=np.uint8)
    # print img1.shape, padding.shape
    return np.concatenate((img1, padding, img2, padding, img3), axis=1)

def main():
    paths = glob("att_faces/s*")
    paths.sort(key=lambda x: int(x.split("s")[-1]))
    imgs = [cv2.imread(path + "/1.pgm") for path in paths]
    top_imgs = []
    for i in xrange(5):
        for j in xrange(i + 1, 5):
            for k in xrange(j + 1, 5):
                concat_img = concat_imgs(imgs[i], imgs[j], imgs[k])
                top_imgs.append((concat_img, (i, j, k)))
    tuples = [(i, j, k) for i in xrange(5, 40) for j in xrange(i + 1, 40) for k in xrange(j + 1, 40)]
    shuffle(tuples)
    shuffle(top_imgs)
    print len(imgs)
    idx_top = np.random.randint(len(top_imgs), size=100)
    for idx, (i, j, k) in enumerate(tuples[:100]):
        concat_img = concat_imgs(imgs[i], imgs[j], imgs[k])
        top_img, (ii, jj, kk) = top_imgs[idx_top[idx]]
        padding = np.zeros((15, top_img.shape[1], 3), dtype=np.uint8)
        # fake_img = np.concatenate((top_img, padding, concat_img), axis=0)
        fake_img = top_img
        cv2.imwrite("test/%d_(s%d_s%d_s%d).jpg" % (idx, ii + 1, jj + 1, kk + 1), fake_img)

if __name__ == "__main__":
    main()
