# Reconstructing-3D-histological-structures-using-machine-learning-AI-algorithms

Oral rehabilitation with dental implant born prostheses is a safe and predictable intervention following tooth loss.  Assessment of bone quality of the implant recipient site prior to or during dental implant placement is paramount for the surgeon, because it influences the long term success of the implants. Histological and microarchitectural analasys of bone core biopsy samples harvested from the implant recipient site is an adequate method to assess bone quality. Concurrently with dental implant surgeries there is an opportunity to monitor the bone micro-architecture and histological properties by virtue of bone biopsy sampling following bone augmentation procedures. The gold standard of bone micro-architectural examinations is histomorphometry, which relies on two-dimensional sections to deduce the spatial properties of the structures.

The goal of this study is to present a methodology to create 3 dimensional structures reconstructed from high resolution histological sections by utilizing convolutional neural networks to annotate the images, discard the folded and/or torn section with an unsupervised information-theory-based clustering algorithm and apply Homography to correct perspective of each image. The resulting models then can be used for 3 dimensional micromorphometry analysis to aid the decision making process of medical professionals.

---

## How to use the scripts

1. export_to_png.py: Exports .png images from the .mrxs slides.
    * set DATA_PATH; FOLDER_NAMES; WORKDIR_PATH accordingly.
    * run
2. tile_images.py: Creates a set of overlapping tiles from the full images.
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * set the size and overlap of the tiles as desired
    * set RECONSTRUCT = True, if you want to test reconstruction as well, otherwise keep it at false
    * run
3. predict.ipynb: Uses a pre-trained U-NET to make predictions on the tiles in batches of 100.
    * source .start_jupyter_gpu.sh
    * set DATA_PATH accordingly.
    * run
4. reconstruct.py: Reconstructs the predicted tiles to fullsized images,with avarageing the overlaps.
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * set the size and overlap of the tiles the same as they were in tile_images.py
    * run
5. threshold.py: Apply thersholding and edge smoothing to the predicted images."
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * run
6. filter_images.iypynb: Filter out the failed biopsies based on the Shanon entorpy of the samples.
    * Create padded images with prepare_for_homography
    * run calculate_sym.py --> get mutual information mx.
7. homography.iypynb: Apply perspective correction on the filtered set of images.
    * Decide wether to only run on masks or on masks and images.
8. rename_and_complete.iypynb: Rename the files according to thickness and complete the missing ones with copies.


## Optional scripts:
### Train U-NET
1. stitch_masks.py:
    Stitch the qupath annotations into full-sized image masks.
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * run
2. tile_training.py:
    Creates a set of overlapping tiles from the full training images.
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * set the size and overlap of the tiles as desired, preferably the same as in tile_images.py
    * set RECONSTRUCT = True, if you want to test reconstruction as well, otherwise keep it at false
    * run
3. train_UNET.ipynb:
    Train U-NET to trenary image segmentation and produce nice demo images.
    * source .start_jupyter_gpu.sh
    * set DATA_PATH accordingly.
    * run
4. reconstruct_training.py:
    Reconstructs the predicted tiles to fullsized images,with avarageing the overlaps.
    * set DATA_PATH; WORKDIR_PATH accordingly.
    * set the size and overlap of the tiles as desired, preferably the same as in tile_training.py
    * run
5. find_thershold.ipynb:
    Find the optimal threshold values (argmax) and smoothing using PR-curve and f-score.
    * source .start_jupyter.sh
    * set DATA_PATH accordingly.
    * run

### Misc.:
* prepare_for_homography.py:
    Helper script, that pads all images to be the same size before performing the homography transformation.
    * color_mode can be used to set the output image as grayscale/color
* calculate_sym.py:
    Script that calculates mutual information similarity between a set of padded images.
* homography_transform.py:
    Helper script, that contains a wrapper around asift keypoint detection and performes perspective correction.
    * Download the source code for ASIFT from here: http://www.ipol.im/pub/art/2011/my-asift/demo_ASIFT_src.tar.gz
    * use return_matrix=True to recover the matrix used for the transformation
