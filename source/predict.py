import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
import pathlib as pathlib

def read_image(x):
    ''' 
    Reads the image stored at the given filepath in a RGB format and normalises the data
    
    Args : 
        x: string, filepath
    Returns:
        image : tensor representing the image
    '''
    
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = tf.convert_to_tensor(image, np.float32)
    return image

def read_mask(y):
    ''' 
    Reads the mask stored at the given filepath in a Grayscale format and normalises the data
    
    Args : 
        y: string, filepath
    Returns:
        mask : tensor representing the image
    '''
    
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def mask_to_3d(mask):
    ''' 
    To make the mask readable by opencv
    
    Args: 
        mask : numpy array
    
    Returns: 
        mask : 
    
    '''
    mask = np.squeeze(mask) #remove empty layers of the image
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0)) #permutes the masks
    return mask

def parse_data(x):
    ''' 
    Reads input image and transfers it to a tensorflow tensor
    
    Args :
        x : numpy array
    
    Returns : 
        x : tensorflow tensor
    '''
    
    x = read_image(x)
    x = tf.image.resize(x,(400, 400)) # set the shape of the tensor object 
    return x

def tf_dataset(x):
    ''' 
    Input is a zip of strings of paths
    
        Args :
            x : zip of strings of paths
        Returns :
            dataset : tensor of tensors representing images
    '''
    dataset = []
    for i in tqdm(range(len(x))) :
        dataset.append(parse_data(x[i]))
        
    dataset = tf.stack(dataset) # This transformation applies map_func to each element of this dataset, and returns a new dataset containing the                                   transformed elements, in the same order as they appeared in the input
    return dataset

def parse(y_pred):
    ''' KEKOI ??? '''
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred

def evaluate_normal(model, x_data, path):
    '''
    Reads the input images and makes a prediction using a given model. 
    It saves the resukts directly in a folder 
    
    Args:   
        model : tensorflow model 
        x_data: tensor of tensors (tensorflow type) representing images 
        path  : string, path of the file where the function is called
        
    Returns: None
    '''
    
    total = []
    for i in tqdm(range(len(x_data)), total=len(x_data)):
        x = x_data[i]
        x = np.expand_dims(x, axis=0)
        y_pred1 = parse(model.predict(x)[0][..., -2])
        y_pred2 = parse(model.predict(x)[0][..., -1])
        
        y_pred2 = tf.image.resize(y_pred2, [608,608])
        h, w, _ = y_pred2.shape

        c = mask_to_3d(y_pred2) * 255.0
        image_filename = path + "/results/result_only" + str(i) + ".png"
        cv2.imwrite(image_filename, c)

def main_predict() :
    '''
    Does the whole prediction process at once : reads the test data, loads and 
    evaluates the model, calls the function which performs the prediction.
    
    Args : None
    Returns : None
    '''
    create_dir("results/")
    path = pathlib.Path(__file__).parent.resolve()
    path = str(path)
    test_path = path + "/test/"
    
    test_x = []
    for i in range(1,51) :
        test_x.append(test_path + "images/test_" + str(i) + ".png")
        
    test_dataset = tf_dataset(test_x)
    test_steps = len(test_x)

    model = load_model_weight("files/model.h5")
    model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_dataset, path)