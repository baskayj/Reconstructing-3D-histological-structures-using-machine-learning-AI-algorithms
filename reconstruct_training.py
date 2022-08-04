import os
from tqdm import tqdm

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import cv2 as cv

# Images we want to tile
DATA_PATH = "Data/training_set/predicted_tiles/"
# Place where the tiles go
WORKDIR_PATH = "Data/training_set/"
# Size of tiles
size = 512
# Overlap between the tiles
overlap = 64

picklenames = []
for picklename in sorted(os.listdir(f"{WORKDIR_PATH}")):
    if picklename.endswith(".pickle"):
        picklenames.append(picklename)

for picklename in tqdm(picklenames):
    df = pd.read_pickle(WORKDIR_PATH + picklename)

    name_str = picklename.replace("_tiles.pickle", "")

    xs = []
    ys = []
    for x, y in df.bottom_right:
        xs.append(x)
        ys.append(y)

    canvas = np.zeros((max(ys), max(xs), 2))
    background = np.ones((max(ys), max(xs), 1))*255
    canvas = np.concatenate((background,canvas), axis = -1)

    for i in range(len(df)):
        img = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[i].n}.png")
        img = np.asarray(img).astype(int)
        top_left = df.loc[i].top_left
        top_right = df.loc[i].top_right
        bottom_left = df.loc[i].bottom_left
        bottom_right = df.loc[i].bottom_right

        # TOP LEFT CORNER
        top_left_corner = img[0:overlap, 0:overlap].copy()
        counter = 1
        # Find any other images, that contain this corner
        for j in range(len(df)):
            if df.loc[j].bottom_right[0] == top_left[0] + overlap and df.loc[j].bottom_right[1] == top_left[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_left_corner += img_2[size - overlap:size, size - overlap:size]
                counter += 1
            elif df.loc[j].bottom_left[0] == top_left[0] and df.loc[j].bottom_left[1] == top_left[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_left_corner += img_2[size - overlap:size, 0:overlap]
                counter += 1
            elif df.loc[j].top_right[0] == top_left[0] + overlap and df.loc[j].top_right[1] == top_left[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_left_corner += img_2[0:overlap, size - overlap:size]
                counter += 1
        img[0:overlap, 0:overlap] = top_left_corner / counter

        # TOP RIGHT CORNER
        top_right_corner = img[0:overlap, size - overlap:size].copy()
        counter = 1
        # Find any other images, that contain this corner
        for j in range(len(df)):
            if df.loc[j].bottom_left[0] == top_right[0] - overlap and df.loc[j].bottom_left[1] == top_right[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_right_corner += img_2[size - overlap:size, 0:overlap]
                counter += 1
            elif df.loc[j].bottom_right[0] == top_right[0] and df.loc[j].bottom_right[1] == top_right[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_right_corner += img_2[size - overlap:size, size - overlap:size]
                counter += 1
            elif df.loc[j].top_left[0] == top_right[0] - overlap and df.loc[j].top_left[1] == top_right[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_right_corner += img_2[0:overlap, 0:overlap]
                counter += 1
        img[0:overlap, size - overlap:size] = top_right_corner / counter

        # BOTTOM LEFT CORNER
        bottom_left_corner = img[size - overlap:size, 0:overlap].copy()
        counter = 1
        # Find any other images, that contain this corner
        for j in range(len(df)):
            if df.loc[j].top_right[0] == bottom_left[0] + overlap and df.loc[j].top_right[1] == bottom_left[1] - overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_left_corner += img_2[0:overlap, size - overlap:size]
                counter += 1
            elif df.loc[j].top_left[0] == bottom_left[0] and df.loc[j].top_left[1] == bottom_left[1] - overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_left_corner += img_2[0:overlap, 0:overlap]
                counter += 1
            elif df.loc[j].bottom_right[0] == bottom_left[0] + overlap and df.loc[j].bottom_right[1] == bottom_left[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_left_corner += img_2[size - overlap:size, size - overlap:size]
                counter += 1
        img[size - overlap:size, 0:overlap] = bottom_left_corner / counter

        # BOTTOM RIGHT CORNER
        bottom_right_corner = img[size - overlap:size, size - overlap:size].copy()
        counter = 1
        # Find any other images, that contain this corner
        for j in range(len(df)):
            if df.loc[j].top_left[0] == bottom_right[0] - overlap and df.loc[j].top_left[1] == bottom_right[1] - overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_right_corner += img_2[0:overlap, 0:overlap]
                counter += 1
            elif df.loc[j].top_right[0] == bottom_right[0] and df.loc[j].top_right[1] == bottom_right[1] - overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_right_corner += img_2[0:overlap, size - overlap:size]
                counter += 1
            elif df.loc[j].bottom_left[0] == bottom_right[0] - overlap and df.loc[j].bottom_left[1] == bottom_right[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_right_corner += img_2[size - overlap:size, 0:overlap]
                counter += 1
        img[size - overlap:size, size - overlap:size] = bottom_right_corner / counter

        # TOP MIDDLE
        top_middle = img[0:overlap, overlap:size - overlap].copy()
        counter = 1
        for j in range(len(df)):
            if df.loc[j].bottom_left[0] == top_left[0] and df.loc[j].bottom_left[1] == top_left[1] + overlap and df.loc[j].bottom_right[0] == top_right[0] and df.loc[j].bottom_right[1] == top_right[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                top_middle += img_2[size - overlap:size, overlap:size - overlap]
                counter += 1
        img[0:overlap, overlap:size - overlap] = top_middle / counter

        # BOTTOM MIDDLE
        bottom_middle = img[size - overlap:size, overlap:size - overlap].copy()
        counter = 1
        for j in range(len(df)):
            if df.loc[j].top_left[0] == bottom_left[0] and df.loc[j].top_left[1] == bottom_left[1] + overlap and df.loc[j].top_right[0] == bottom_right[0] and df.loc[j].top_right[1] == bottom_right[1] + overlap:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                bottom_middle += img_2[0:overlap, overlap:size - overlap]
                counter += 1
        img[size - overlap:size, overlap:size - overlap] = bottom_middle / counter

        # LEFT MIDDLE
        left_middle = img[overlap:size - overlap, 0:overlap].copy()
        counter = 1
        for j in range(len(df)):
            if df.loc[j].top_right[0] == top_left[0] + overlap and df.loc[j].top_right[1] == top_left[1] and df.loc[j].bottom_right[0] == bottom_left[0] + overlap and df.loc[j].bottom_right[1] == bottom_left[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                left_middle += img_2[overlap:size - overlap, size - overlap:size]
                counter += 1
        img[overlap:size - overlap, 0:overlap] = left_middle / counter

        # RIGHT MIDDLE
        right_middle = img[overlap:size - overlap, size - overlap:size].copy()
        counter = 1
        for j in range(len(df)):
            if df.loc[j].top_left[0] == top_right[0] + overlap and df.loc[j].top_left[1] == top_right[1] and df.loc[j].bottom_left[0] == bottom_right[0] + overlap and df.loc[j].bottom_left[1] == bottom_right[1]:
                img_2 = cv.imread(f"{DATA_PATH}{name_str}_{df.loc[j].n}.png")
                img_2 = np.asarray(img_2).astype(int)
                right_middle += img_2[overlap:size - overlap, 0:overlap]
                counter += 1
        img[overlap:size - overlap, size - overlap:size] = right_middle / counter

        # Adding the corrected image to the canvas
        xmin = top_left[0]
        xmax = top_right[0]
        ymin = top_left[1]
        ymax = bottom_left[1]
        canvas[ymin:ymax, xmin:xmax] = img

        cv.imwrite(f"{WORKDIR_PATH}predicted_reconstructed/{name_str}.png", canvas)
