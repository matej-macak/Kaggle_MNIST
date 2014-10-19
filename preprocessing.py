import pandas as pd
import numpy as np
import skimage.transform as tf
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os

import time

class FontGen():
    """
    This is a class that generates new data using fonts stored on the computer. 
    The list self.non_fonts stores values for fonts which do not correspond to
    numbers and is rejected. 
    """    
    
    def __init__(self):
        self.non_fonts = (('vrindab.ttf'),
             ('webdings.ttf'),
             ('wingding.ttf'),
             ('WINGDNG2.TTF'),
             ('WINGDNG3.TTF'),
             ('ZWAdobeF.TTF'),
             ('symbol.ttf'),
             ('REFSPCL.TTF'),
             ('marlett.ttf'),
             ('OUTLOOK.TTF'),
             ('BSSYM7.TTF'))
        
        self.path = '/WINDOWS/Fonts/'
        self.list_fonts()
        self.generate_images(self.fonts)        
        
    def list_fonts(self):
        """
        Stores list all font file names on the system (those starting with .ttf or .TTF) in
        fonts attribute.
        """    
        for root, dirs, files in os.walk(self.path):
            self.fonts= [fl for fl in files if '.ttf' in fl or '.TTF' in fl]
            
    def generate_images(self,fonts,path_dir='/WINDOWS/Fonts/', non_fonts = []):
        """ 
        Generate MNIST-like images from the font library on a computer. Optionally
        pass in non_fonts variable which specifies font name (file name without 
        the directory that do not correspond to numbers. Stores normalized numpy
        array in images attribute.
        """
        
        self.images = []
        for number in xrange(10):
            #Do this for all the numbers    
            
            for font_base in fonts:
                if (font_base) not in self.non_fonts:
                    # Create font path
                    font_path = path_dir+font_base
                    
                    # Generate the image
                    base_img = Image.fromarray(np.uint8(np.zeros((28,28))))
                    draw = ImageDraw.Draw(base_img)
                    font = ImageFont.truetype(str(font_path),22)
                    draw.text((14,0),str(number), font=font,fill=255)
        
                    # Normalize and stack the image into an array
                    arr = (np.array(base_img,dtype=np.float32)/255.0).reshape((1,784))
                    if self.images == []:
                        self.images = arr
                        self.labels = np.array(number)
                    else:
                        self.images = np.vstack((self.images,arr))
                        self.labels = np.vstack((self.labels,number))
        

def load_data(path, case = 'train'):
    """Takes the path to the csv file and returns X, y values (for the train set).
    The values are normalized to 0-1.0 and converted to float32 for Theano
    GPU compatibility.
    """    
    
    if case == 'train':
        data = pd.read_csv(path).values
        X = np.float32(data[:,1:])/255.0
        y = np.int64(data[:,0])
    elif case == 'test':
        data = np.float32(pd.read_csv(path).values)/255.0
    
    return X, y

def initialize(data, img_shape):
    """
    Helper function to save some code to create arrays to be transformed.
    """
    shape = data.shape
    reshaped = data.reshape((shape[0],img_shape[0],img_shape[1]))
    new = np.float32(np.zeros((shape[0],img_shape[0],img_shape[1])))
    
    return shape, reshaped, new
    
def batch_transforms(data, img_shape = (28,28), scale = True, scale_interval = (0.9,1.1),
                     shear = True, shear_interval = (-0.05,0.05),
                     rotate = True, rotate_interval = (-7.5,7.5)):
                         
    """Applies various transforms (shear, scale, rotate) using uniformly sampled 
    values on the pre-determined interval. 
    """
    
    shape, reshaped, new = initialize(data,img_shape)    
    for loc in xrange(shape[0]):
        
        tf_image = reshaped[loc,:,:]
        
        #Perform scaling
        if scale == True:
            scale_x, scale_y = np.random.uniform(low = scale_interval[0],
                                                 high = scale_interval[1],
                                                 size = (2,1))
            tf_image = tf.warp(tf_image,tf.AffineTransform(scale=(scale_x[0],scale_y[0])))
            
        #Perform shearing
        if shear == True:
            shear_val = np.random.uniform(low=shear_interval[0],high=shear_interval[0])
            tf_image = tf.warp(tf_image,tf.AffineTransform(shear=shear_val))
        
        #Perform rotating
        if rotate == True:
            rotate_val = np.random.uniform(low=rotate_interval[0],high = rotate_interval[1])
            tf_image = tf.rotate(tf_image,rotate_val)
            
        new[loc,:,:] = tf_image
        
    new = new.reshape((shape[0],img_shape[0]*img_shape[1]))
    return new
    







