import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def main():
    imgs = [cv2.imread(path + "/1.pgm") for path in glob("att_faces/s[1-3]")]
    fake_img = np.concatenate(imgs, axis=1)
    plt.imshow(fake_img, cmap="gray")
    plt.show()
    cv2.imwrite("test1.jpg", fake_img)

if __name__ == "__main__":
    main()
