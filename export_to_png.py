import os
from tqdm import tqdm

import numpy as np

import cv2 as cv
import openslide

from math import ceil,floor

# This is where the scans live, change it if something happens
DATA_PATH = "../../../../../media/san2_ssd/data/csont_biopszia_copy/"
# These are the folders inside, instead of listing them it might be easier to always just use those, that are needed!
FOLDER_NAMES = ["E_fixed/", "F/"]

# Path to save the images
WORKDIR_PATH = "Data/source_files/"

for foldername in FOLDER_NAMES:
    # Find the scans
    filenames = []
    for filename in sorted(os.listdir(f"{DATA_PATH}{foldername}")):
        if filename.endswith(".mrxs"):
            filenames.append(filename)

    for filename in tqdm(filenames):
        #print(f"{DATA_PATH}{foldername}{filename}")
        slide = openslide.OpenSlide(f"{DATA_PATH}{foldername}{filename}")

        img = slide.read_region((0, 0), (slide.level_count - 1),
                                slide.level_dimensions[(slide.level_count - 1)])  # get the slide
        img = np.array(img)  # to numpy array
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)  # fix the color channels
        # Fix the alpha channel
        alpha_channel = img[:, :, 3]
        _, mask = cv.threshold(alpha_channel, 254, 255, cv.THRESH_BINARY)  # binarize mask
        color = img[:, :, :3]
        img = cv.bitwise_not(cv.bitwise_not(color, mask=mask))
        # Make a grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edged = cv.Canny(img, 100, 255)
        # Apply adaptive threshold
        thresh = cv.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
        thresh_color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
        thresh = cv.dilate(thresh, None, iterations=15)
        thresh = cv.erode(thresh, None, iterations=15)
        # Find the contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        X = 0
        Y = 0
        W = 0
        H = 0
        armax = 0
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w*h > armax:
                cnt_max = cnt
                armax = w*h
                X = x
                Y = y
                W = w
                H = h
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
        if not armax == 0:
            # starting the export
            outfile = filename.replace(".mrxs", "")
            #print(outfile)

            level = 3
            scale_1 = int(slide.level_downsamples[(slide.level_count - 1)])
            scale_2 = int(scale_1 / slide.level_downsamples[level])

            real_box = (box - [X, Y]) * scale_2  # the real size params of the rotated rect
            real_cnt = (cnt_max - [X,Y]) * scale_2 # the real size params of the rotated ROI

            high_res = slide.read_region((X * scale_1, Y * scale_1), level, (W * scale_2, H * scale_2))
            high_res = np.array(high_res)  # to numpy array
            high_res = cv.cvtColor(high_res, cv.COLOR_RGBA2BGRA)  # fix the color channels
            alpha_channel = high_res[:, :, 3]
            _, mask = cv.threshold(alpha_channel, 254, 255, cv.THRESH_BINARY)  # binarize mask
            color = high_res[:, :, :3]
            high_res = cv.bitwise_not(cv.bitwise_not(color, mask=mask))

            #high_res = cv.drawContours(high_res,[real_box],0,(0,0,255),50)

            # How big the padding should be:
            if min(real_box[:, 1]) < 0:
                top = (-1) * min(real_box[:, 1])
            else:
                top = 0
            if max(real_box[:, 1]) > np.shape(high_res)[0]:
                bottom = max(real_box[:, 1]) - np.shape(high_res)[0]
            else:
                bottom = 0
            if min(real_box[:, 0]) < 0:
                left = (-1) * min(real_box[:, 0])
            else:
                left = 0
            if max(real_box[:, 0]) > np.shape(high_res)[1]:
                right =max(real_box[:, 0]) - np.shape(high_res)[1]
            else:
                right = 0
            high_res = cv.copyMakeBorder(high_res, top,
                                         bottom, left,
                                         right, cv.BORDER_CONSTANT,
                                         value=[255, 255, 255])  # add padding to fit in the whole rotated rectangle

            new_box = real_box + [left, top]
            new_cnt = real_cnt + [left, top]
            high_res_roi = high_res.copy()
            high_res_roi = cv.fillPoly(np.zeros(np.shape(high_res_roi)), pts =[new_cnt], color=(255,255,255))
            #high_res = cv.drawContours(high_res, [new_box], 0, (0, 0, 255), 50)

            new_rect = cv.minAreaRect(new_box)
            center = (new_rect[0][0], new_rect[0][1])

            # Getting the right angle of rotation
            if new_rect[2] > 5:
                theta = new_rect[2]-90
                width = new_rect[1][1]  # the 90 degrees switches height and width
                height = new_rect[1][0]  # the 90 degrees switches height and width
            else:
                theta = new_rect[2]
                width = new_rect[1][0]
                height = new_rect[1][1]

            #print(f"center:{center}")
            #print(f"width:{width}")
            #print(f"height:{height}")

            # Rotating the biopsy
            shape = (high_res.shape[1], high_res.shape[0])
            matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
            high_res = cv.warpAffine(src=high_res, M=matrix, dsize=shape, borderMode=cv.BORDER_CONSTANT,
                                   borderValue=[255, 255, 255])
            high_res_roi = cv.warpAffine(src=high_res_roi, M=matrix, dsize=shape, borderMode=cv.BORDER_CONSTANT,
                                   borderValue=[0, 0, 0])
            fx = 0.85
            fy = 0.96
            top = ceil(np.shape(high_res_roi)[0]*(1-fy)*0.5)
            bottom = floor(np.shape(high_res_roi)[0]*(1-fy)*0.5)
            left = ceil(np.shape(high_res_roi)[1]*(1-fx)*0.5)
            right = floor(np.shape(high_res_roi)[1]*(1-fx)*0.5)
            high_res_roi = cv.resize(high_res_roi,(ceil(np.shape(high_res_roi)[1]*fx),ceil(np.shape(high_res_roi)[0]*fy)))
            high_res_roi = cv.copyMakeBorder(high_res_roi, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
            # Slice only what's needed
            x_min = int(center[0] - width / 2)
            if x_min < 0:
                x_min = 0
            x_max = int(x_min + width)
            if x_max > np.shape(high_res)[1]:
                x_max = np.shape(high_res)[1]
            y_min = int(center[1] - height / 2)
            if y_min < 0:
                y_min = 0
            y_max = int(y_min + height)
            if y_max > np.shape(high_res)[0]:
                y_max = np.shape(high_res)[0]

            #print(np.shape(high_res))
            #print(f"x_min:{x_min}")
            #print(f"x_max:{x_max}")
            #print(f"y_min:{y_min}")
            #print(f"y_max:{y_max}")

            high_res = high_res[y_min:y_max, x_min:x_max]
            high_res_roi = high_res_roi[y_min:y_max, x_min:x_max]
            _,high_res_roi= cv.threshold(high_res_roi, 128, 255, cv.THRESH_BINARY)
            cv.imwrite(f"{WORKDIR_PATH}{outfile}.png",high_res)
            cv.imwrite(f"{WORKDIR_PATH}../roi/{outfile}.png",high_res_roi)
            #print("----------------------")
