import os
from tqdm import tqdm
from math import ceil

import numpy as np
import pandas as pd

import cv2 as cv

# Images we want to tile
DATA_PATH = "Data/training_set/source_files/"
# Place where the tiles go
WORKDIR_PATH = "Data/training_set/"
# Size of tiles
size = 512
# Overlap between the tiles
overlap = 64

# Reconstruct the tiles for debugging
RECONSTRUCT = True



# First we tile the masks
filenames = []
for filename in sorted(os.listdir(f"{DATA_PATH}masks/")):
    if filename.endswith(".png"):
        filenames.append(filename)

print("Creating Mask Tiles\n")
for filename in tqdm(filenames):
    img = cv.imread(f"{DATA_PATH}masks/{filename}", 0)
    img = np.asarray(img)

    filename = filename.replace(".png", "")

    n = 0 # The number of the tile
    outfile = [] # File containing metadata for easy reconstruction
    x_size = ceil((np.shape(img)[1]/size)*(1+overlap/size))
    y_size = ceil((np.shape(img)[0]/size)*(1+overlap/size))
    for i in range(x_size):
        # i goes through x
        for j in range(y_size):
            # j goes through y
            tile = img[j*(size-overlap):j*(size-overlap)+size,i*(size-overlap):i*(size-overlap)+size]
            if not (np.shape(tile)[0] == size and np.shape(tile)[1] == size):
                tile = cv.copyMakeBorder(tile,0,size-np.shape(tile)[0],0,size-np.shape(tile)[1],cv.BORDER_CONSTANT,value=[255,255,255])
            unique_pixels = np.unique(tile.flatten())
            if not len(unique_pixels) == 1 and not unique_pixels[0] == 255:
                top_left = [i*(size-overlap),j*(size-overlap)]
                top_right = [i*(size-overlap)+size,j*(size-overlap)]
                bottom_left = [i*(size-overlap),j*(size-overlap)+size]
                bottom_right = [i*(size-overlap)+size,j*(size-overlap)+size]
                file_name = filename + f"_{n}.png"
                cv.imwrite(f"{WORKDIR_PATH}mask_tiles/{file_name}", tile)
                outfile.append([n,top_left,top_right,bottom_left,bottom_right,file_name])
            n += 1

    # Saving the metadata
    outfile = pd.DataFrame(data=outfile,columns=["n","top_left","top_right","bottom_left","bottom_right","file_name"])
    outfile.to_pickle(f"{WORKDIR_PATH}{filename}_tiles.pickle")



# Then we tile the images in the same way
print("Creating Image Tiles\n")
for filename in tqdm(filenames):
    img = cv.imread(f"{DATA_PATH}images/{filename}")
    img = np.asarray(img)

    filename = filename.replace(".png", "")

    picklename = filename + "_tiles.pickle"
    df = pd.read_pickle(WORKDIR_PATH + picklename)

    n = 0  # The number of the tile
    x_size = ceil((np.shape(img)[1]/size)*(1+overlap/size))
    y_size = ceil((np.shape(img)[0]/size)*(1+overlap/size))
    for i in range(x_size):
        # i goes through x
        for j in range(y_size):
            # j goes through y
            tile = img[j * (size - overlap):j * (size - overlap) + size, i * (size - overlap):i * (size - overlap) + size]
            if not (np.shape(tile)[0] == size and np.shape(tile)[1] == size):
                tile = cv.copyMakeBorder(tile,0,size-np.shape(tile)[0],0,size-np.shape(tile)[1],cv.BORDER_CONSTANT,value=[255,255,255])
            if n in list(df.n):
                file_name = filename + f"_{n}.png"
                cv.imwrite(f"{WORKDIR_PATH}image_tiles/{file_name}", tile)
            n += 1

print("Reconstructing Masks and Images\n")
if RECONSTRUCT:
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

        subfolders = ["mask_tiles/", "image_tiles/"]
        for subfolder in subfolders:
            canvas = np.ones((max(ys), max(xs))) * 255

            name_str = subfolder + name_str
            for i in range(len(df)):
                img = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[i].n}.png", 0)
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
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        top_left_corner += img_2[size - overlap:size, size - overlap:size]
                        counter += 1
                    elif df.loc[j].bottom_left[0] == top_left[0] and df.loc[j].bottom_left[1] == top_left[1] + overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        top_left_corner += img_2[size - overlap:size, 0:overlap]
                        counter += 1
                    elif df.loc[j].top_right[0] == top_left[0] + overlap and df.loc[j].top_right[1] == top_left[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
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
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        top_right_corner += img_2[size - overlap:size, 0:overlap]
                        counter += 1
                    elif df.loc[j].bottom_right[0] == top_right[0] and df.loc[j].bottom_right[1] == top_right[1] + overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        top_right_corner += img_2[size - overlap:size, size - overlap:size]
                        counter += 1
                    elif df.loc[j].top_left[0] == top_right[0] - overlap and df.loc[j].top_left[1] == top_right[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
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
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_left_corner += img_2[0:overlap, size - overlap:size]
                        counter += 1
                    elif df.loc[j].top_left[0] == bottom_left[0] and df.loc[j].top_left[1] == bottom_left[1] - overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_left_corner += img_2[0:overlap, 0:overlap]
                        counter += 1
                    elif df.loc[j].bottom_right[0] == bottom_left[0] + overlap and df.loc[j].bottom_right[1] == bottom_left[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
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
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_right_corner += img_2[0:overlap, 0:overlap]
                        counter += 1
                    elif df.loc[j].top_right[0] == bottom_right[0] and df.loc[j].top_right[1] == bottom_right[1] - overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_right_corner += img_2[0:overlap, size - overlap:size]
                        counter += 1
                    elif df.loc[j].bottom_left[0] == bottom_right[0] - overlap and df.loc[j].bottom_left[1] == bottom_right[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_right_corner += img_2[size - overlap:size, 0:overlap]
                        counter += 1
                img[size - overlap:size, size - overlap:size] = bottom_right_corner / counter

                # TOP MIDDLE
                top_middle = img[0:overlap, overlap:size - overlap].copy()
                counter = 1
                for j in range(len(df)):
                    if df.loc[j].bottom_left[0] == top_left[0] and df.loc[j].bottom_left[1] == top_left[1] + overlap and df.loc[j].bottom_right[0] == top_right[0] and df.loc[j].bottom_right[1] == top_right[1] + overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        top_middle += img_2[size - overlap:size, overlap:size - overlap]
                        counter += 1
                img[0:overlap, overlap:size - overlap] = top_middle / counter

                # BOTTOM MIDDLE
                bottom_middle = img[size - overlap:size, overlap:size - overlap].copy()
                counter = 1
                for j in range(len(df)):
                    if df.loc[j].top_left[0] == bottom_left[0] and df.loc[j].top_left[1] == bottom_left[1] + overlap and df.loc[j].top_right[0] == bottom_right[0] and df.loc[j].top_right[1] == bottom_right[1] + overlap:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        bottom_middle += img_2[0:overlap, overlap:size - overlap]
                        counter += 1
                img[size - overlap:size, overlap:size - overlap] = bottom_middle / counter

                # LEFT MIDDLE
                left_middle = img[overlap:size - overlap, 0:overlap].copy()
                counter = 1
                for j in range(len(df)):
                    if df.loc[j].top_right[0] == top_left[0] + overlap and df.loc[j].top_right[1] == top_left[1] and df.loc[j].bottom_right[0] == bottom_left[0] + overlap and df.loc[j].bottom_right[1] == bottom_left[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
                        img_2 = np.asarray(img_2).astype(int)
                        left_middle += img_2[overlap:size - overlap, size - overlap:size]
                        counter += 1
                img[overlap:size - overlap, 0:overlap] = left_middle / counter

                # RIGHT MIDDLE
                right_middle = img[overlap:size - overlap, size - overlap:size].copy()
                counter = 1
                for j in range(len(df)):
                    if df.loc[j].top_left[0] == top_right[0] + overlap and df.loc[j].top_left[1] == top_right[1] and df.loc[j].bottom_left[0] == bottom_right[0] + overlap and df.loc[j].bottom_left[1] == bottom_right[1]:
                        img_2 = cv.imread(f"{WORKDIR_PATH}{name_str}_{df.loc[j].n}.png", 0)
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

            name_str = name_str.replace(subfolder, "")
            cv.imwrite(f"{WORKDIR_PATH}{subfolder}reconstructed/{name_str}.png", canvas)
