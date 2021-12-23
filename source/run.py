import data as data
from glob import glob
import model as Model
import train as Train
import trainTuning as tT
import predict as Predict
import pathlib
import mask_to_submission as m2s
import pandas as pd
import numpy as np
import predict as Predict
import cv2


    #data augmentation
#data.augment_split()
   
    #model training
#Train.main_train()
    
Predict.main_predict() 

m2s.main_mask_to_submission()
