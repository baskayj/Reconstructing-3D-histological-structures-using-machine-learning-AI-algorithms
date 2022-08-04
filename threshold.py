import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tqdm

DATA_PATH = "Data/"
scale = 2

ids = next(os.walk(f"{DATA_PATH}predicted_reconstructed"))[2]
print("No. of images = ", len(ids))

for i in tqdm.tqdm(ids):
    img = cv.imread(f"{DATA_PATH}predicted_reconstructed/{i}")
    w = np.shape(img)[0]
    h = np.shape(img)[1]
    img = cv.resize(cv.bilateralFilter(cv.resize(img,None,fx = 1/scale,fy = 1/scale, interpolation = cv.INTER_CUBIC),-1,50,50),(h,w),fx = scale,fy = scale, interpolation = cv.INTER_CUBIC)      # Bilateral Smoothing on predicted masks
    img_t = 1-(np.argmax(img, axis = 2))/2
    cv.imwrite(f"{DATA_PATH}predicted_thresholded/{i}", img_t*255)
    # plt.imshow(img_t, "gray")
    # plt.show()