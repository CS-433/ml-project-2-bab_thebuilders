# The source folder 
This folder contains all the scripts we used to create, build and train our model, that are then used by the script `run.py`.

- `data.py`: contains the functions used to preprocess our dataset, to do the data augmentation and the split between training and validation set
- `metrics.py` : contains the implementation of the different losses we used at some point in our project
- `model.py` : contains the definition of the double UNet
- `predict.py` : contains the functions used to predict the segmentation of the images of the test set, using a trained model
- `train.py` : contains the functions used to train the model
- `trainTuning.py` : contains the functions used to tune hyperparameters
- `utils.py` : contains every helper function we used at some point, such as reading images or loading paths
- `mask_to_submission.py` : contains the functions to turn the output of the model into the submission file
- `submission_to mask.py` : contains the functions to turn the submission file to masks to have a visual 
