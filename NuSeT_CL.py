import PIL.Image, PIL.ImageTk
from skimage.transform import rescale
import numpy as np
import os
import tensorflow as tf
from test import test, test_single_img, test_UNet, test_single_img_UNet
from train_gui import train_NuSeT, train_UNet 

class NuSeT_CL:
    def __init__(self):
        self.params = {}
        # Default values, adjust them to have better performance

        # Normalization method, fg = foreground normalization
        #                       wn = whole image normalization
        self.params['normalization_method'] = 'fg'

        ################ Inference parameters #################
        self.params['watershed'] = 'yes'
        self.params['min_score'] = 0.95
        self.params['nms_threshold'] = 0.15
        self.params['postProcess'] = 'yes'
        self.params['scale_ratio'] = 1.0
        # Specify the testing img directories here
        self.batch_seg_path = './sample_image/'

        ################ Training parameters #################
        self.params['lr'] = 5e-5
        self.params['optimizer'] = 'rmsprop'
        self.params['epochs'] = 35
        # Currently can choose to train UNet or NuSeT
        self.params['model'] = 'NuSeT'
        # Specify the training img and label directories here
        self.train_img_path = './path_to_training_imgs/'
        self.train_label_path = './path_to_training_labels/'
        self.usingCL = True

    def train(self):

        # By default, we use NuSeT to train our model
        if self.params['model'] == 'NuSeT':
            if self.params['normalization_method'] == 'fg':
                # Train with whole image norm for the first round
                self.params['normalization_method'] = 'wn'
                with tf.Graph().as_default():
                    train_NuSeT(self)
                
                # Train with foreground normalization for the second round
                self.params['normalization_method'] = 'fg'
                with tf.Graph().as_default():
                    train_NuSeT(self)
            
            else:
                self.params['normalization_method'] = 'wn'
                with tf.Graph().as_default():
                    train_NuSeT(self)

        # Train with U-Net
        else:
            if self.params['normalization_method'] == 'fg':
                # Train with whole image norm for the first round
                self.params['normalization_method'] = 'wn'
                with tf.Graph().as_default():
                    train_UNet(self)
                
                # Train with foreground normalization for the second round
                self.params['normalization_method'] = 'fg'
                with tf.Graph().as_default():
                    train_UNet(self)
            
            else:
                self.params['normalization_method'] = 'wn'
                with tf.Graph().as_default():
                    train_UNet(self)

    def segmentation_batch(self):
        if self.params['model'] == 'NuSeT':
            with tf.Graph().as_default():
                test(self.params, self)
        else:
            with tf.Graph().as_default():
                test_UNet(self.params, self)

    def fix_img_dimension(self):
        self.height = self.height//16*16
        self.width = self.width//16*16
        self.im_np = self.im_np[:self.height, :self.width]

nuset = NuSeT_CL()

# Specify the action here, enable one of the lines to perform training or 
# testing. For example, I have enabled only testing here
#nuset.train()
nuset.segmentation_batch()
