# The source folder 
This folder contains all the scripts we used to create, build and train our model, that are then used by the script `run.py`.

- `data.py`: contains the functions used to preprocess our dataset, to do the data augmentation and the split between training and validation set
- `learningrateTuning.ipynb` : notebook containing the functions and plots used to tune the learning rate
- `mask_to_submission.py` : contains the functions to turn the output of the model into the submission file
- `metrics.py` : contains the implementation of the different losses we used at some point in our project
- `model.py` : contains the definition of the double UNet
- `predict.py` : contains the functions used to predict the segmentation of the images of the test set, using a trained model
- `submission_to mask.py` : contains the functions to turn the submission file to masks to have a visual (purely helper functions)
- `train.py` : contains the functions used to train the model
- `trainTuning.py` : contains the functions used to tune hyperparameters
- `train_with_FL` : notebook containing the training steps with the focal loss, that we didn't use in the end
- `utils.py` : contains every helper function we used at some point, such as reading images or loading paths


### To run :
The script `run.py` calls functions written in other scripts to train the model (if necessary) and predict the outputs on the test set.
