import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from tqdm import tqdm

DATA_PATH = "Data/training_set/qupath_annotations/"
WORKDIR_PATH = "Data/training_set/source_files/"

ids = next(os.walk(f"{WORKDIR_PATH}images"))[2]

for i in tqdm(ids):
    source_img = cv.imread(f"{WORKDIR_PATH}images/{i}", 0)
    source_img = np.asarray(source_img)

    name_str = i.replace(".png", "")
    #print(name_str)

    canvas = np.zeros((np.shape(source_img)[0], np.shape(source_img)[1]))

    # Collecting all patches corresponding to a given image
    patches = next(os.walk(f"{DATA_PATH}"))[2]
    tmp = []
    list(map(lambda x: tmp.append(x) if x.startswith(name_str) else "do nothing", patches))
    patches = tmp
    del tmp

    for patch in patches:
        patch_cut = patch.replace(name_str, "").replace(")-mask.png", "")
        if "_Tumor_(" in patch_cut:
            patch_cut = patch_cut.replace("_Tumor_(", "")
            mask_type = 1
        elif "_Stroma_(" in patch_cut:
            patch_cut = patch_cut.replace("_Stroma_(", "")
            mask_type = 2
        elif "_Necrosis_(" in patch_cut:
            patch_cut = patch_cut.replace("_Necrosis_(","")
            mask_type = 3

        scale_factor = int(float(patch_cut.split(",")[0]))
        x_upper_left = int(patch_cut.split(",")[1])
        y_upper_left = int(patch_cut.split(",")[2])
        x_size = int(patch_cut.split(",")[3])
        y_size = int(patch_cut.split(",")[4])

        img = cv.imread(f"{DATA_PATH}{patch}", 0)
        img = cv.resize(img, (x_size, y_size), scale_factor, scale_factor)
        img = (img > 0.5).astype(np.uint8)
        img = np.asarray(img)

        for j in range(y_size):
            for k in range(x_size):
                if canvas[y_upper_left + j, x_upper_left + k] == 0 or canvas[y_upper_left + j, x_upper_left + k] == 255:
                    if img[j, k] == 1:
                        canvas[y_upper_left + j, x_upper_left + k] = img[j, k] * mask_type * 0.25 * 255
                    else:
                        canvas[y_upper_left + j, x_upper_left + k] = 255

    for j in range(np.shape(canvas)[0]):
        for k in range(np.shape(canvas)[1]):
            if canvas[j,k] == 0:
                canvas[j,k] = 255

    #plt.imshow(source_img, "gray")
    #plt.imshow(canvas/255, alpha=0.6)
    #plt.show()

    cv.imwrite(f"{WORKDIR_PATH}masks/{name_str}.png", canvas)


