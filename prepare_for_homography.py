import os
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
from math import floor,ceil

def prepare_images(img_list,source_path,target_path,color_mode = False,padding = 500,border_value=(255,255,255)):
    print("Finding optimal canvas size.")
    max_h = 0
    max_w = 0
    for i in tqdm(img_list):
        img = cv.imread(f'{source_path}{i}',0)
        h = np.shape(img)[1]
        w = np.shape(img)[0]
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
    print("Padding images.")
    for i in tqdm(img_list):
        if color_mode:
            img = cv.imread(f'{source_path}{i}')
        else:
            img = cv.imread(f'{source_path}{i}',0)
        h = np.shape(img)[1]
        w = np.shape(img)[0]
        left = ceil((max_h + padding*2 - h)/2)
        right = floor((max_h + padding*2 - h)/2)
        top = ceil((max_w + padding*2 - w)/2)
        bottom = floor((max_w + padding*2 - w)/2)
        dst = cv.copyMakeBorder(img,top,bottom,left,right,cv.BORDER_CONSTANT,value = border_value)
        cv.imwrite(f'{target_path}{i}',dst)
    print('Done.')