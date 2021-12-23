# README
This is the repository containing the proposed solution of the team _BAB-thebuilders_ to the problem of road segmentation that was proposed in the scope in the Machine Learning course (CS-433) from EPFL during the fall semester of 2021. Our team proposed a deep learning approach based on a double UNet accompanied with various innovative solutions. We managed to achieve a F1-score of __0.844__ and a precision of __0.916__.

## Team Members
- Bastien Golomer (PH-MA3)
- Bastien Le Lan (SC-MA1)
- Arthur Brousse (SC-MA1)

## Repository description
- The dataset is available on the aicrowd.com page [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files).
- The bulk of our python functions are in the folder **source**. Detailed explanations are available inside it.

## Pre-trained model
The pre-trained model is too voluminous to be stored on GitHub. Instead, it is available [here](https://drive.google.com/drive/folders/1VrlhBvzDyHwNom-jpgtSr4l86XyrQc1W?usp=sharing). 

The path for the model has to be source/files/model.h5. The test files also need to be in a specific path: /source/test/images/test_"image_id".png

The path for the training data is : /source/training_sat_images/images/satImage_"image_id".png for the images and /source/training_sat_images/masks/satImage_"mask_id".png for the masks.

## How to run the code 
Note that the provided files do not allow one to reproduce our _best_ submission on AICrowd, since the file containing the corresponding model has been lost. The provided files allow one to reproduce the second best submission we made (ID 169569), which had an F-1 score of __0.834__ and a precision of __0.910__.

To reproduce the prediction, one needs to run the script run.py with the correct architecture (mentionned above). To do the data augmentation and train the model one simply needs to decomment the lines in run.py: 

-data.augment_split()

-Train.main_train()
