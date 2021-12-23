import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
from utils import *
import pathlib
from PIL import Image
import imutils
import random


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    ToGray,
    RandomRotate90,
    Rotate,
    RandomSizedCrop,
    RandomBrightnessContrast,
    RandomGamma,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
#    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
#    CoarseDropout
)

from cv2 import(
    cvtColor,
    GaussianBlur,
    Sobel
    )

    
def rotate_patches(x,y):
    ''' Inputs are PIL images, outputs are np.arrays'''

    batch_size = 200
    # split the image and mask in 4 parts 
    #   ---------
    #   | 1 | 2 |
    #   ---------
    #   | 3 | 4 | 
    #   ---------
    x = np.array(x)
    y = np.array(y)
    
    
    quarter_1_x = x[:200,:200]
    quarter_1_y = y[:200,:200]
    quarter_2_x = x[:200,200:]
    quarter_2_y = y[:200,200:]
    quarter_3_x = x[200:,:200]
    quarter_3_y = y[200:,:200]
    quarter_4_x = x[200:,200:]
    quarter_4_y = y[200:,200:]
    
    aug = RandomRotate90(p=1)
    augmented = aug(image=quarter_1_x, mask=quarter_1_y)
    x1 = augmented['image']
    y1 = augmented['mask']
    
    augmented = aug(image=quarter_2_x, mask=quarter_2_y)
    x2 = augmented['image']
    y2 = augmented['mask']
    
    augmented = aug(image=quarter_3_x, mask=quarter_3_y)
    x3 = augmented['image']
    y3 = augmented['mask']
    
    augmented = aug(image=quarter_4_x, mask=quarter_4_y)
    x4 = augmented['image']
    y4 = augmented['mask']
       
    x1 = Image.fromarray(x1)
    y1 = Image.fromarray(y1)
    x2 = Image.fromarray(x2)
    y2 = Image.fromarray(y2)
    x3 = Image.fromarray(x3)
    y3 = Image.fromarray(y3)
    x4 = Image.fromarray(x4)
    y4 = Image.fromarray(y4)
    
    
    image = Image.new('RGB', (400,400),color = 'black') 
    image.paste(x1,(0,0))
    image.paste(x2,(0,batch_size-1))
    image.paste(x3,(batch_size-1,0))
    image.paste(x4,(batch_size-1,batch_size-1))

    mask = Image.new('RGB', (400,400),color = 'black') 
    mask.paste(y1,(0,0))
    mask.paste(y2,(0,batch_size-1))
    mask.paste(y3,(batch_size-1,0))
    mask.paste(y4,(batch_size-1,batch_size-1))
    image, mask =  np.array(image),np.array(mask)

    return image, mask
   
def rotation_padding(x,y) :
    ''' Inputs are numpy arrays, outputs are np.arrays'''
    
    angleListe = [-90,0,90,180]
    size = 400, 400
    
    # COMMENT : in order to do the rotation and paste, we need to switch the image mode to RGBA
    x = Image.fromarray(x)
    y = Image.fromarray(y)

    # Rotation  pi/6, random rotation for the background
    angle = random.choice(angleListe)
    dst_im = x.convert('RGBA') 
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = x.convert('RGBA')
    rot = im.rotate( 30, expand=1 ).resize(size) # pi/6
    dst_im.paste( rot, (0, 0), rot )
    x5 = dst_im

    dst_im = y.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = y.convert('RGBA')
    rot = im.rotate( 30, expand=1 ).resize(size) # pi/6
    dst_im.paste( rot, (0, 0), rot )
    y5 = dst_im
    
    # Rotation  pi/4, random rotation for the background
    angle = random.choice(angleListe)
    dst_im = x.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = x.convert('RGBA')
    rot = im.rotate( 45, expand=1 ).resize(size) # pi/4
    dst_im.paste( rot, (0, 0), rot )
    x6 = dst_im
    
    dst_im = y.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = y.convert('RGBA')
    rot = im.rotate( 45, expand=1 ).resize(size) # pi/4
    dst_im.paste( rot, (0, 0), rot )
    y6 = dst_im
    
    
    # Rotation  pi/3, random rotation for the background
    angle = random.choice(angleListe)
    dst_im = x.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = x.convert('RGBA')
    rot = im.rotate( 60, expand=1 ).resize(size) # pi/3
    dst_im.paste( rot, (0, 0), rot )
    x7 = dst_im
    
    dst_im = y.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = y.convert('RGBA')
    rot = im.rotate( 60, expand=1 ).resize(size) # pi/3
    dst_im.paste( rot, (0, 0), rot )
    y7 = dst_im
    
    
    
    # Rotation  pi/2
    x8 = x.rotate(90)
    y8 = y.rotate(90)
    
    # Rotation  pi/4, random rotation for the background
    angle = random.choice(angleListe)
    dst_im = x.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = x.convert('RGBA')
    rot = im.rotate( 120, expand=1 ).resize(size) # 2 pi/3
    dst_im.paste( rot, (0, 0), rot )
    x9 = dst_im
    
    dst_im = y.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = y.convert('RGBA')
    rot = im.rotate( 120, expand=1 ).resize(size) # 2 pi/3
    dst_im.paste( rot, (0, 0), rot )
    y9 = dst_im
    
    # Rotation  5 pi/6, random rotation for the background
    angle = random.choice(angleListe)
    dst_im = x.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = x.convert('RGBA')
    rot = im.rotate( 160, expand=1 ).resize(size) # 5 pi/6
    dst_im.paste( rot, (0, 0), rot )
    x10 = dst_im
    
    dst_im = y.convert('RGBA')
    dst_im = dst_im.rotate( angle, expand=1 ).convert("RGBA")
    im = y.convert('RGBA')
    rot = im.rotate( 160, expand=1 ).resize(size) # 5 pi/6
    dst_im.paste( rot, (0, 0), rot )
    y10 = dst_im
    
    # convert back to RGB
    x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10= x5.convert('RGB'),y5.convert('RGB'),x6.convert('RGB'),y6.convert('RGB'),x7.convert('RGB'),y7.convert('RGB'), x8.convert('RGB'), y8.convert('RGB'), x9.convert('RGB'), y9.convert('RGB'),x10.convert('RGB'), y10.convert('RGB')
    
    x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10 = np.array(x5),np.array(y5),np.array(x6),np.array(y6),np.array(x7),np.array(y7),np.array(x8),np.array(y8),np.array(x9),np.array(y9),np.array(x10),np.array(y10)

    return x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10

def reduce_multiply(x,y, number) : 
    ''' Inputs are np.arrays, outputs np.arrays '''
    
    def solve(n):
       sq_root = int(np.sqrt(n))
       return (sq_root*sq_root) == n
    if solve(number) == False :
        raise failedException
        
    size = len(x)   
    size = int(len(x)/np.sqrt(number))
    x = Image.fromarray(x)
    y = Image.fromarray(y)
    resized_x = x.resize((size,size))
    resized_y = y.resize((size,size))

    image = Image.new('RGB', (400,400),color = 'black') 
    mask = Image.new('RGB', (400,400),color = 'black') 

    
    for i in range(number):
        for j in range(number):
                image.paste(resized_x,(i*(size-1),j*(size-1)))
                mask.paste(resized_y,(i*(size-1),j*(size-1)))
            
    image = image.resize((400,400))
    mask = mask.resize((400,400))

    return [np.array(image), np.array(mask)]
    
def blur(img,fact):
    img3=ndimage.gaussian_filter(img, sigma=fact)
    return img3

def switcharoo(img):
    img1=img.copy()
    img1[:,:,0]=img[:,:,2]
    img1[:,:,1]=img[:,:,0]
    img1[:,:,2]=img[:,:,1]
    img2=img.copy()
    img2[:,:,0]=img[:,:,1]
    img2[:,:,1]=img[:,:,2]
    img2[:,:,2]=img[:,:,0]
    return img1,img2

def saturation(img,factor):
    img4=Image.fromarray(img)
    converter = ImageEnhance.Color(img4)
    img4 = converter.enhance(factor)
    img4=np.array(img4)
    return img4

def lightness(img, factor):
    img2=Image.fromarray(img)
    enhancer=ImageEnhance.Brightness(img2)
    img_light = enhancer.enhance(factor)
    img2=np.array(img_light)
    return img2

def sharpness(img, factor):
    img2=Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img2)
    img2=enhancer.enhance(factor)
    img2=np.array(img2)
    return img2

def rotation_zoom(img,angle):
    taille=len(img[:,0,0])
    zoom=round((taille*(math.sin(math.radians(angle))+math.cos(math.radians(angle))))/taille,3)+0.001
    img1=ndimage.zoom(img,(zoom,zoom,1))
    img1=ndimage.rotate(img1, angle,reshape=False)
    centre=(len(img1[:,0,0])/2)
    gauche=int(centre-taille/2)
    droite=int(centre+taille/2)
    img1=img1[gauche:droite,gauche:droite,:]
    return img1

def shear_vertical(img1,factor):
    H, W = img1[:,:,0].shape
    zoom=round(1/(1-factor),3)
    M2 = np.float32([[1, 0, 0], [factor, 1, 0]])
    M2[0,2] = -M2[0,1] * W/2
    M2[1,2] = -M2[1,0] * H/2
    aff2 = cv2.warpAffine(img1, M2, (W, H))
    gauche=int(H/2-H/2*(1-factor))
    droite=int(H/2+H*(1-factor)/2)
    aff2=aff2[gauche:droite,:,:]
    aff2=ndimage.zoom(aff2,(zoom,1,1))
    return aff2

def shear_horizontal(img1,factor):
    H, W = img1[:,:,0].shape
    zoom=round(1/(1-factor),3)
    M2 = np.float32([[1, factor, 0], [0, 1, 0]])
    M2[0,2] = -M2[0,1] * W/2
    M2[1,2] = -M2[1,0] * H/2
    aff2 = cv2.warpAffine(img1, M2, (W, H))
    gauche=int(H/2-H/2*(1-factor))
    droite=int(H/2+H*(1-factor)/2)
    aff2=aff2[:,gauche:droite,:]
    aff2=ndimage.zoom(aff2,(1,zoom,1))
    return aff2

def boing(img,fact,Vertical):
    vert=Vertical
    class WaveDeformer:
        def transform(self, x, y):
            if vert:
                x = x + fact*math.sin(x/40)
            else:
                y = y + fact*math.sin(x/40)
            return x, y
        def transform_rectangle(self, x0, y0, x1, y1):
            return (*self.transform(x0, y0),
                    *self.transform(x0, y1),
                    *self.transform(x1, y1),
                    *self.transform(x1, y0),
                    )
        def getmesh(self, img):
            self.w, self.h = img.size
            gridspace = 20
            target_grid = []
            for x in range(0, self.w, gridspace):
                for y in range(0, self.h, gridspace):
                    target_grid.append((x, y, x + gridspace, y + gridspace))

            source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

            return [t for t in zip(target_grid, source_grid)]
    img4=Image.fromarray(img)
    result_image = ImageOps.deform(img4, WaveDeformer())
    img2=np.array(result_image)
    return img2
 
def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. 
    Input : 
        * images = tuple of path strings
        * masks = tuple of path strings
        * save path = string
        * augment = bool, decides wether to do the augmentation or not"""
    crop_size = (400-100, 400-100)
    size = (400, 400)

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        #isolate the name of the images and masks
        image_name = image.split("/")[-1].split(".")[0]
        mask_name = mask.split("/")[-1].split(".")[0]

        
        x, y = read_data(image, mask)
        try:
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = read_data(image, mask)
            h, w, c = x.shape

        if augment == True:


            ## Random Rotate 90 degree
            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            ## Transpose
            aug = Transpose(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']
            
            # rotation + pasting over a random rotation of the original image
            x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8, = rotation_padding(x,y)

            
            # random rotation quarter patches 
            augmented = rotate_patches(x , y)
            x9 = augmented[0]
            y9 = augmented[1]
            
            # random rotation quarter patches 
            augmented = rotate_patches(x , y)
            x10 = augmented[0]
            y10 = augmented[1]
            
            
            augmented = reduce_multiply(x,y, number = 4)
            augmented = aug(image=x, mask=y)
            x11 = augmented['image']
            y11 = augmented['mask']

            augmented = reduce_multiply(x,y,9)
            x12 = augmented[0]
            y12 = augmented[1]

            augmented = reduce_multiply(x,y, number = 16)
            x13 = augmented[0]
            y13 = augmented[1]
            
            x14,y14,_,_,_,_,x15,y15,_,_,x16,y16 = rotation_padding(x11,y11)

            
            ## Grayscale
            aug = ToGray(p=1)
            augmented = aug(image=x, mask=y)
            x17 = augmented['image']
            y17 = augmented['mask']

            ## Vertical Flip gray
            augmented = aug(image=x17, mask=y17)
            x18 = augmented['image']
            y18 = augmented['mask']

            ## Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x17, mask=y17)
            x19 = augmented['image']
            y19= augmented['mask']


            ## rotate the grayimage
            _,_,_,_,x20,y20,_,_,x21,y21,_,_ = rotation_padding(x17,y17)
            
            ## translate images
            x22 = np.roll(x, (50,50), axis=(0,1))
            y22 = np.roll(y, (50,50), axis=(0,1))
            
            ## translate images
            x23 = np.roll(x, (-50,-50), axis=(0,1))
            y23 = np.roll(y, (-50,50), axis=(0,1))
            
            ## blur with gaussian filter
            x24 = blur(x,2)
            x25 = blur(x,4)
            y24 = y
            y25 = y

            ## higher and lower saturation
            x26=saturation(x,2)
            y26=y
            x27=saturation(x,0.5)
            y27=y

            ##higher and lower brightness
            x28=lightness(x,2.5)
            y28=y
            x29=lightness(x,0.5)
            y29=y

            ## switch the RGB columns
            x30,x31= switcharoo(x)
            y30=y
            y31=y

            ## change the image sharpness
            x32=sharpness(x,5)
            y32=y
            x33=sharpness(x,-4)
            y33=y

            ##vertical shear its possible to add .8 but the image is very deformed
            x33=shear_vertical(x,0.2)
            x34=boing(x,10, True)
            x35=boing(x,30,True)

            y33=shear_vertical(y,0.2)
            y34=boing(x,10, True)
            y35=boing(x,30,True)

             ##horizontal shearits possible to add .8 but the image is very deformed
            x37=shear_horizontal(x,0.2)
            x38=boing(x,10, False)
            x36=boing(x,30,False)

            y37=shear_horizontal(y,0.2)
            y38=boing(x,10, False)
            y36=boing(x,30,False)

            ##rotation zoom many rotation angle possible between 0 and 90
            x41=rotation_zoom(x,20)
            y41=rotation_zoom(y,20)

            x42=rotation_zoom(x,40)
            y42=rotation_zoom(y,40)

            x39=rotation_zoom(x,60)
            y39=rotation_zoom(y,60)

            x40=rotation_zoom(x,80)
            y40=rotation_zoom(y,80)

            #the augmentations fro x24 to x42 did not bring significant improvement and were therefore not used for this model
            # build the images and masks arrays
            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                x21,x22,x23 #,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,
                #x34,x35,x36,x37,x38,x39,x40,x41,x42
            ]
            masks  = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
                y21,y22,y23#,y24,y25,y26,y27,y28,y29,y30,y31,y32,y33,
                #y34,y35,y36,y37,y38,y39,y40,y41,y42
            ]

        else:
            images = [x]
            masks  = [y]

        idx = 0
        for i in range(len(images)):
            image = images[i]
            mask = masks[i]
            # image_name defined in the previous for loop
            tmp_image_name = f"{image_name}_{idx}.png"
            tmp_mask_name  = f"{mask_name}_{idx}.png"

            # define the path to where we should save the image
            image_path = os.path.join(save_path, "images/", tmp_image_name)
            mask_path  = os.path.join(save_path, "masks/", tmp_mask_name)

            # save the images at the previous path
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            image.save(image_path)
            mask.save(mask_path)

            idx += 1

def load_data(path, split=0.1):
    """ Load all the data from the specified path and then split them into train and valid dataset. """

    image_list = []

    img_path = glob(str(path) + '/training_sat_images/images/*.png')
    msk_path = glob(str(path) + '/training_sat_images/masks/*.png')

    img_path.sort()
    msk_path.sort()

    len_ids = len(img_path)
    train_size = int((1.0-split)*len_ids)
    valid_size = int(split*len_ids)		## Here 10 is the percent of images used for validation

    # splits the data into training and validation sets
    train_x, valid_x = train_test_split(img_path, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(msk_path, test_size=valid_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y) # these are just the paths

def get_split_data(path, split=0.1):

    train_x = glob(os.path.join(path, "trainx/*"))
    train_y = glob(os.path.join(path, "trainy/*"))

    valid_x = glob(os.path.join(path, "validationx/*"))
    valid_y = glob(os.path.join(path, "validationy/*"))

    return (train_x, train_y), (valid_x, valid_y)
    
    
    
## main function for data augmentation

def augment_split(path = pathlib.Path(__file__).parent.resolve()):
    ''' This function does the whole augmentation process at once.
    It selects the current path as the default path '''
    np.random.seed(42)
    path = str(path)
    (train_x, train_y), (valid_x, valid_y) = load_data(path, split=0.1) # loads the paths

    create_dir(path + "/new_data/train/images/")
    create_dir(path + "/new_data/train/masks/")
    create_dir(path + "/new_data/valid/images/")
    create_dir(path + "/new_data/valid/masks/")


    augment_data(train_x, train_y, path + "/new_data/train/", augment=True)
    augment_data(valid_x, valid_y, path + "/new_data/valid/", augment=True)
    # augment_data(test_x, test_y, "new_data/test/", augment=False)

