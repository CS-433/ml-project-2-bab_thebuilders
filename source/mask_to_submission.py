#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import pathlib as pathlib
from tqdm import tqdm
from PIL import Image

def check_corner(input_matrix) :
# define all the corners
    SW_corner = [[1,0,0],
                [1,0,0],
                [1,1,1]]
    SW_corner = np.array(SW_corner)
    SE_corner = [[0,0,1],
                [0,0,1],
                [1,1,1]]
    SE_corner = np.array(SE_corner)            
    NE_corner = [[1,1,1],
                [0,0,1],
                [0,0,1]]     
    NE_corner = np.array(NE_corner)
    NW_corner = [[1,1,1],
                [1,0,0],
                [1,0,0]]  
    NW_corner = np.array(NW_corner)

    corner = np.array_equal(input_matrix, SW_corner) or np.array_equal(input_matrix, SE_corner) or np.array_equal(input_matrix, NE_corner) or np.array_equal(input_matrix, NW_corner)
    
    return corner

foreground_threshold = 0.05 # percentage of pixels > 1 required to assign a foreground label to a patch


def fill(im, reverted):
    threshold = 4
    bad = 0  # value that we want to get rid of
    good = 1  # value that we want
    if reverted == True:  # if the matrix has been inverted, we need to interchange the values
        bad = 1
        good = 0

    N = len(im)
    # added to take into account the border conditions
    background_matrix = np.zeros((N+2, N+2))
    # fill the background matrix
    background_matrix[0, :] = good
    background_matrix[N+1, :] = good
    background_matrix[:, 0] = good
    background_matrix[:, N+1] = good

    background_matrix[1:N+1, 1:N+1] = im
    im = background_matrix
    change_matrix =  np.zeros((N+2, N+2))


    for i in range(1, N+2):
        for j in range(1, N+2):
            surrounding_matrix = im[i-1:i+2, j-1:j+2]
            if im[i, j] == bad and not(check_corner(surrounding_matrix)):
                
                somme = np.sum(surrounding_matrix)
                change_matrix[i,j] = somme

                if somme > threshold :
                        im[i, j] = good

    return im[1:N+1,1:N+1]



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
    reduced_size = im.shape[0]/patch_size
    print(reduced_size)
    reduced_image = np.zeros((int(reduced_size),int(reduced_size)))
    
    # read the image and fill the black squares
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch,foreground_threshold)
            reduced_image[int(i/patch_size),int(j/patch_size)] = label
            
    image = fill(reduced_image, reverted = False)
            
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            yield("{:03d}_{}_{},{}".format(img_number +1, j, i, int(image[int(i/patch_size),int(j/patch_size)])))


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
