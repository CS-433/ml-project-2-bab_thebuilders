#!/usr/bin/python
import os
import sys
from PIL import Image
import math
import matplotlib.image as mpimg
import numpy as np

label_file = 'submission.csv'

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
# imgwidth = int(math.ceil((400.0/w))*w)
# imgheight = int(math.ceil((400.0/h))*h)
nc = 3


def binary_to_uint8(img):
    '''
    Converts an array of binary labels to a uint8
    
    Args:
        img : matrix of floats
    Returns:
        rimg: matrix of unint8
    '''
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id):
    ''' 
    Reads the representation of the i'th image (of the predictions of the model) in the submission file,
    from which it creates an image, stored in a specified folder
    
    Args:
        image_id: int, index of the image in the list of images
    
    Returns:
        im : PIL image, representation of the mask after it has been passed through the mask_to_sumbission method
    '''
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im

def main_submission2mask(): 
    ''' 
    Does the whole reconstruction of the masks after they have been simplified to fit the submission format of AI crowd.
    It ranges over all the test images.
    '''
    for i in range(0, 50):
        reconstruct_from_labels(i)
   
