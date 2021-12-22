import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_model
from utils import *
from metrics import *
import pathlib


def read_image(x):
    ''' 
    Reads the image stored at the given filepath in a RGB format and normalises the data
    
    Args : 
        x: string, filepath
    Returns:
        image : cv2 array
    '''
    x = x.decode() # string method, decodes the string using the codec registered for encoding. It defaults to the default string encoding.
    image = cv2.imread(x, cv2.IMREAD_COLOR) # loads the RGB image itself from the path x
    image = np.clip(image - np.median(image) + 127, 0, 255) # Given an interval, values outside the interval are clipped to the interval bounds.
    image = image/255.0 # the values of the image matrix are now in interval [0,1]
    image = image.astype(np.float32) # transformation for later modification
    return image

def read_mask(y):
    ''' 
    Reads the mask stored at the given filepath in a Grayscale format and normalises the data
    
    Args : 
        y: string, filepath
    Returns:
        mask : cv2 array representing the image
    '''
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) # loads the RGB image itself from the path x
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1) # Insert a new axis that will appear at the axis position in the expanded array shape : (400,400,1)
    return mask

def parse_data(x, y):
    ''' 
    Reads input images and transfers it to tensorflow tensors
    
    Args :
        x : tensor
        y : tensor
    
    Returns : 
        x : tensorflow tensor
        y : tensorflow tensor

    '''
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1) # tranformation to have twice the size for later processes (400,400,2)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([400, 400, 3]) # set the shape of the tensor object 
    y.set_shape([400, 400, 2])
    return x, y

def tf_dataset(x, y, batch=8):
    '''
    Args : 
        x :     list of strings, representing the filepaths
        y :     list of strings, representing the filepaths
        batch : Combines consecutive elements of this dataset into batches
    Returns : 
        dataset : tensorflow dataset
    '''
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) # Creates a Dataset whose elements are slices of the given tensors. The given tensors are     sliced along their first dimension. This operation preserves the structure of the input tensors, removing the first dimension of each tensor       and using it as the dataset dimension. All input tensors must have the same size in their first dimensions.
    
    dataset = dataset.shuffle(buffer_size=32) # tf.data.Dataset.shuffle() method randomly shuffles a tensor along its first dimension.
    dataset = dataset.map(map_func=parse_data) # This transformation applies map_func to each element of this dataset, and returns a new dataset                                                    containing the transformed elements, in the same order as they appeared in the input
    dataset = dataset.repeat() # Repeats this dataset so each original value is seen count times. The default behavior (if count is None or -1) is                                     for the dataset be repeated indefinitely.
    dataset = dataset.batch(batch) 

    return dataset

def main_train():
    ''' 
    Does the whole training process at once
    It fetches the data (can be either augmented or not) and builds a model using the DoubleUnet architecture
    The model is directly saved in a .h5
    
    Args :      None
    Returns :   None
    '''
    
    np.random.seed(42)
    tf.random.set_seed(42)
    path = pathlib.Path(__file__).parent.resolve()
    path = str(path)
    create_dir(path + "/files")

    train_path = path + "/new_data/train/"
    valid_path = path + "/new_data/valid/"

    ## fetch all the paths of all the images
    train_x = glob( train_path + "images/*.png")
    train_y = glob(train_path + "masks/*.png")

    # give the same order to the images and the masks
    train_x.sort()
    train_y.sort()

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Validation
    valid_x = sorted(glob( valid_path + "images/*.png"))
    valid_y = sorted(glob(valid_path + "masks/*.png"))

    model_path = path + "/files/model.h5"

    # hyperparameters
    batch_size = 7
    epochs = 10
    lr = 1e-4
    shape = (400, 400, 3)

    # fetching the model from the model.py file
    model = build_model(shape)
    metrics = [
        dice_coef,
        Recall(),
        Precision()
    ]
    
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    model.compile(loss = focal_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]

    train_steps = (len(train_x)/batch_size)
    valid_steps = (len(valid_x)/batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=True) # here the model is stored in the "files/model.h5" which can be fetched later using predict.py
