import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

smooth = 1e-15
def dice_coef(y_true, y_pred):
    ''' 
    Computes the Dice coefficient, aka the F1 score.
    It is a statistic used to gauge the similarity of two samples

    Args :
        y_true : array, contains the true values
        y_pred : array, contains the predicted value computed with the model
        
    Returns:
        dice_coeff : float
    '''
    
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    ''' 
    Computes the Dice Loss, which optimizes the Dice coefficient
    It is the most commonly used segmentation evaluation metric.
    
    Args :
        y_true : array, contains the true values
        y_pred : array, contains the predicted value computed with the model
            
    Returns:
        dice_loss : float
    '''
    
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    '''
    iou = Intersection over Union is an evaluation metric used to measure 
    the accuracy of an object detector on a particular dataset. 

    Args :
        y_true : array, contains the true values
        y_pred : array, contains the predicted value computed with the model
            
    Returns:
        dice_loss : float    
    '''
    
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def bce_dice_loss(y_true, y_pred):
    '''
    Binary cross entropy + dice loss

    Args :
        y_true : array, contains the true values
        y_pred : array, contains the predicted value computed with the model
            
    Returns:
        bce_loss + dice_loss : float    
    '''
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def focal_loss(y_true, y_pred):
    ''' 
    Computes the focal loss
    Args :
        y_true : array, contains the true values
        y_pred : array, contains the predicted value computed with the model
            
    Returns:
        focal loss : float    
    '''
    
    alpha=0.25
    gamma=2
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)
