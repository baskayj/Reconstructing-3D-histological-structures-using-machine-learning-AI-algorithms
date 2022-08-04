import os
import numpy as np
import cv2 as cv
import pickle

from tqdm import tqdm
from image_similarity_measures.quality_metrics import ssim

DATA_PATH = os.getcwd() + '/Data/'

def mutual_information(hgram):
    """ Mutual information for joint histogram"""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

        
print("Working on the benchmark images")
ids = next(os.walk(f"{DATA_PATH}predicted_padded/raw/"))[2]
print("No. of images = ", len(ids))

img_list = ids[:180]

minf_sym = np.zeros((len(img_list),(len(img_list))))
for i in tqdm(range(len(img_list))):
    src = cv.imread(f'{DATA_PATH}predicted_padded/raw/{img_list[i]}',0)
    for j in range(i,len(img_list)):
        dst = cv.imread(f'{DATA_PATH}predicted_padded/raw/{img_list[j]}',0)
        hist_2d, _, _ = np.histogram2d(src.flatten(),dst.flatten(),bins=20)
        minf_sym[i,j] = mutual_information(hist_2d)
        outfile = open(f"{DATA_PATH}minf_sym_raw.pickle","wb")
        pickle.dump(minf_sym, outfile)
        outfile.close()
