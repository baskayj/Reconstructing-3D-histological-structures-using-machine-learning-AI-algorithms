import os
import subprocess
import cv2 as cv
import pandas as pd
import numpy as np

def ASIFT_matcher(src_path,dst_path):
    os.chdir(f'{os.getcwd()}/demo_ASIFT_src')
    proc = subprocess.Popen(['./demo_ASIFT', src_path, dst_path,'../imgOutVert.png','../imgOutHori.png','../matchings.txt','../keys1.txt','../keys2.txt'])
    proc.wait()
    os.chdir(f'{os.getcwd()}/..')


def Homography_Transform(src,dst,scale = 2,return_matrix = False):
    #src is the source image we want to tarnsform
    #dst is the target, we want the source to look like
    h = np.shape(dst)[1]
    w = np.shape(dst)[0]

    src_scaled = cv.resize(src,None,fx=1/scale,fy=1/scale,interpolation = cv.INTER_CUBIC)
    dst_scaled = cv.resize(dst,None,fx=1/scale,fy=1/scale,interpolation = cv.INTER_CUBIC)

    cv.imwrite(f'{os.getcwd()}/src.png',src_scaled)
    cv.imwrite(f'{os.getcwd()}/dst.png',dst_scaled)

    ASIFT_matcher(f'{os.getcwd()}/src.png',f'{os.getcwd()}/dst.png')
    df = pd.read_csv("matchings.txt",skiprows = 1, header = None, sep = '  ', engine='python')
    p1 = np.zeros((len(df),2),dtype = np.float32)
    p2 = np.zeros((len(df),2),dtype = np.float32)
    for i in range(len(df)):
        p1[i,0] = df.iloc[i,0]
        p1[i,1] = df.iloc[i,1]
        p2[i,0] = df.iloc[i,2]
        p2[i,1] = df.iloc[i,3]

    #Calculate the Homography matrix
    H, status = cv.findHomography(p1,p2,cv.RANSAC,5.0)
        
    #Scale the Homography matrix
    H[0,2] = scale*H[0,2]
    H[1,2] = scale*H[1,2]
    H[2,0] = (1/scale)*H[2,0]
    H[2,1] = (1/scale)*H[2,1]

    #Transformation
    src_warped = cv.warpPerspective(src, H, (dst.shape[1],dst.shape[0]),borderMode=cv.BORDER_CONSTANT,borderValue=(255,255,255))
    
    #Binarize Image
    _,thresh = cv.threshold(src_warped, 128, 255, cv.THRESH_BINARY)
    _,thresh2 = cv.threshold(255-src_warped, 254, 255, cv.THRESH_BINARY)
    _,thresh3 = cv.threshold(255-thresh2-thresh, 128, 128, cv.THRESH_BINARY)
    src_binary = thresh3 + thresh
    
    #Cleanup
    os.remove('imgOutVert.png')
    os.remove('imgOutHori.png')
    os.remove('matchings.txt')
    os.remove('keys1.txt')
    os.remove('keys2.txt')
    os.remove(f'{os.getcwd()}/src.png')
    os.remove(f'{os.getcwd()}/dst.png')
    
    if return_matrix:
        return np.array(src_binary),H
    else:
        return np.array(src_binary)
    
    
def Homography_Transform_From_Matrix(src,dst,H,border_value=(255,255,255)):
    src_warped = cv.warpPerspective(src, H, (dst.shape[1],dst.shape[0]),borderMode=cv.BORDER_CONSTANT,borderValue=border_value)
    return np.array(src_warped)