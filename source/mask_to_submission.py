#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import pathlib as pathlib
from tqdm import tqdm


foreground_threshold = 0.05 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch,foreground_threshold):
    ''' 
    Decides whether or not a patch should be labeled 0 or 1
    
    Args: 
        patch:                  matrix, batch of pixels
        forerground_threshold : float, the threshold at which the batch is labeled 1
    Returns:
        int, the value of the pixel
    
    '''
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename,i,foreground_threshold):
    """
    Reads a single image and outputs the strings that should go into the submission file to represent that image
    The pixels of the input image are grouped by batches of 16x16 and the foreground_threshold value gives
    the value at which a batch is considered a white pixel.
    
    Args: 
        image_filename          : string, self-explanatory
        i                       : int, index of the image selected in the list of images
        foreground_threshold    : float, gives the threshold of what is considered a white pixel
                                         in the mask->submission transformation
        
    Returns: None
    """
    
    img_number = i
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch,foreground_threshold)
            yield("{:03d}_{}_{},{}".format(img_number +1, j, i, int(label)))


def masks_to_submission(submission_filename, image_filenames,foreground_threshold = 0.25):
    """
    From a series of images filenames, creates and fills directly a submission file compatible with the online platform AIcrowd.
    
    Args:
        submission_filename : string, self explanatory
        image_filenames     : list of strings, paths of the images
        
    Returns: None
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i,fn in enumerate(image_filenames):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn,i,foreground_threshold))


def main_mask_to_submission():
    '''
    Does the whole process of creating the submission on one go. 
    It reads all the masks predicted and fills the submission appropriately.
    
    Args : None
    Returns : None
    '''
    
    submission_filename = 'submission.csv'
    image_filenames = []
    
    path = pathlib.Path(__file__).parent.resolve()
    path = str(path)
    test_path = path + "/results/"
    submission_filename = path + '/' + submission_filename
    
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
    
    for i in tqdm(range(50)):
        
        image_filename = test_path + 'result_only' + str(i) + '.png'
        image_filenames.append(image_filename)
    print(image_filenames)
    masks_to_submission(submission_filename, image_filenames)
