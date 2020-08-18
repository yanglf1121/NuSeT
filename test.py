import tensorflow as tf
import numpy as np

import csv

from PIL import Image
from tqdm import tqdm
from skimage.transform import rescale

from model_layers.models import UNET
from model_layers.model_RPN import RPN
from model_layers.anchor_size import anchor_size
from model_layers.rpn_target import RPNTarget
from model_layers.rpn_proposal import RPNProposal
from model_layers.rpn_loss import RPNLoss
from model_layers.seg_loss import segmentation_loss
from model_layers.marker_watershed import marker_watershed
from model_layers.compute_metrics import compute_metrics

from utils.load_data import load_data_test
from utils.tf_utils import optimizer_fun
from utils.anchors import generate_anchors_reference
from utils.generate_anchors import generate_anchors
from utils.test import generate_gt_boxes
from utils.normalization import whole_image_norm, foreground_norm, clean_image
from utils.losses import smooth_l1_loss
from utils.image_vis import draw_rpn_bbox_pred, draw_gt_boxes, draw_top_nms_proposals, draw_rpn_bbox_targets, draw_rpn_bbox_pred_only 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# inspired from https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/rpn_test.py
def test(params, self):
    """Predict masks for all images in a given directory, and save them  

    Args:
        params (dict): the parameters of the network
    """
    
    # Get the testing parameters 
    perform_watershed = params['watershed']
    bbox_min_score = params['min_score'] 
    nms_thresh = params['nms_threshold']
    postProcess = params['postProcess']
    resize_scale = params['scale_ratio']

    # Load the data
    # x_test, y_test: test images and corresponding labels
    x_id, x_test = load_data_test(self.batch_seg_path)
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (?,hf,wf,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(?,h,w,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)
    
    # Dynamic anchor base size calculated from median cell lengths
    base_size = anchor_size(tf.reshape(pred_masks,[input_height,input_width]))

    # scales and ratios are used to generate different anchors 
    scales = np.array([ 0.5, 1, 2])
    ratios = np.array([ 0.125, 0.25, 0.5, 1, 2, 4, 8])
    
    # stride is to control how sparse we want to place anchors across the image
    # stride = 16 means to place an anchor every 16 pixels on the original image
    stride = 16
    
    # Generate the anchor reference with respect to the original image
    ref_anchors = generate_anchors_reference(base_size, ratios, scales)
    num_ref_anchors = scales.shape[0] * ratios.shape[0]

    feat_height = input_height / stride
    feat_width = input_width / stride
    
    # Generate all the anchors based on ref_anchors
    all_anchors = generate_anchors(ref_anchors, stride, [feat_height,feat_width])
       
    num_anchors = all_anchors.shape[0]
    with tf.variable_scope('model_RPN') as scope:
        prediction_dict = RPN(feat_map, num_ref_anchors)
    
    # Get the tensors from the dict
    rpn_cls_prob = prediction_dict['rpn_cls_prob']
    rpn_bbox_pred = prediction_dict['rpn_bbox_pred']
    
    proposal_prediction = RPNProposal(rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape, nms_thresh)
    
    pred_dict_final['all_anchors'] = tf.cast(all_anchors, tf.float32)
    prediction_dict['proposals'] = proposal_prediction['proposals']
    prediction_dict['scores'] = proposal_prediction['scores']
        
    pred_dict_final['rpn_prediction'] = prediction_dict
    scores = pred_dict_final['rpn_prediction']['scores']
    proposals = pred_dict_final['rpn_prediction']['proposals']

    pred_masks_watershed = tf.to_float(marker_watershed(scores, proposals, pred_masks, min_score = bbox_min_score))
    

    # start point for testing, and end point for graph 
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    num_batches_test = len(x_test) 

    saver = tf.train.Saver()
    
    masks1 = []
    # Restore the per-image normalization model from the trained network
    saver.restore(sess,'./Network/whole_norm.ckpt')
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        # whole image normalization
        batch_data = x_test[j]   
        batch_data_shape = batch_data.shape
        image = np.reshape(batch_data, [batch_data_shape[0],batch_data_shape[1]])

        if resize_scale != 1:
            image = rescale(image, self.params['scale_ratio'], anti_aliasing=True)

        # Clip the height and width to be 16-fold 
        imheight, imwidth = image.shape
        imheight = imheight//16*16
        imwidth = imwidth//16*16
        image = image[:imheight, :imwidth]

        image_normalized_wn = whole_image_norm(image)
        image_normalized_wn = np.reshape(image_normalized_wn, [1,imheight,imwidth,1])
        
        
        masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_wn})
        if not self.usingCL:
            self.progress_var.set(j/2/num_batches_test*100)
            self.window.update()
            
        # First pass, get the coarse masks, and normalize the image on masks
        masks1.append(masks)
    
    # Restore the foreground normalization model from the trained network
    saver.restore(sess,'./Network/foreground.ckpt')
    
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        batch_data = x_test[j]
        batch_data_shape = batch_data.shape
        image = np.reshape(batch_data, [batch_data_shape[0],batch_data_shape[1]])

        if resize_scale != 1:
            image = rescale(image, self.params['scale_ratio'])
        
        # Clip the height and width to be 16-fold 
        imheight, imwidth = image.shape
        imheight = imheight//16*16
        imwidth = imwidth//16*16
        image = image[:imheight, :imwidth]

        # Final pass, foreground normalization to get final masks
        image_normalized_fg = foreground_norm(image, masks1[j])
        image_normalized_fg = np.reshape(image_normalized_fg, [1,imheight,imwidth,1])
        
        # If adding watershed, we save the watershed masks separately
        if perform_watershed == 'yes':
            
            masks_watershed = sess.run(pred_masks_watershed, feed_dict={train_initial:image_normalized_fg})

            if postProcess == 'yes':
                masks_watershed = clean_image(masks_watershed)

            # Revert the scale to original display
            if resize_scale != 1:
                masks_watershed = rescale(masks_watershed, 1/self.params['scale_ratio'])
            
            I8 = (((masks_watershed - masks_watershed.min()) / (masks_watershed.max() - masks_watershed.min())) * 255).astype(np.uint8)
            img = Image.fromarray(I8)
            img.save(self.batch_seg_path + x_id[j] + '_masks_watershed.png')

        else:
            
            masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_fg})

            if postProcess == 'yes':
                masks = clean_image(masks)

            # enable these 2 lines if your want to see the detection result
            #image_pil = draw_top_nms_proposals(pred_dict, batch_data, min_score=bbox_min_score, draw_gt=False)
            #image_pil.save(str(j)+'_pred.png')

            # Revert the scale to original display
            if resize_scale != 1:
                masks = rescale(masks, 1/self.params['scale_ratio'])

            I8 = (((masks - masks.min()) / (masks.max() - masks.min())) * 255).astype(np.uint8)
            img = Image.fromarray(I8)
            img.save(self.batch_seg_path + x_id[j] + '_masks.png')
        if not self.usingCL:
            self.progress_var.set(50 + j/2/num_batches_test*100)
            self.window.update()
    sess.close()

# This function is similar to the function above, but only for one image that is 
# displayed on NuSeT GUI
def test_single_img(params, x_test):
    """input the image, return the segmented mask

    Args:
        params (dict): the parameters of the network
        x_test: the input image in numpy array
    """
    
    # Get the testing parameters 
    perform_watershed = params['watershed']
    bbox_min_score = params['min_score'] 
    nms_thresh = params['nms_threshold']
    postProcess = params['postProcess']
    
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (?,32,32,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(?,512,512,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)
    
    # Dynamic anchor base size calculated from median cell lengths
    base_size = anchor_size(tf.reshape(pred_masks,[input_height,input_width]))

    # scales and ratios are used to generate different anchors 
    scales = np.array([ 0.5, 1, 2])
    ratios = np.array([ 0.125, 0.25, 0.5, 1, 2, 4, 8])
    
    # stride is to control how sparse we want to place anchors across the image
    # stride = 16 means to place an anchor every 16 pixels on the original image
    stride = 16
    
    # Generate the anchor reference with respect to the original image
    ref_anchors = generate_anchors_reference(base_size, ratios, scales)
    num_ref_anchors = scales.shape[0] * ratios.shape[0]

    feat_height = input_height / stride
    feat_width = input_width / stride
    
    # Generate all the anchors based on ref_anchors
    all_anchors = generate_anchors(ref_anchors, stride, [feat_height,feat_width])
       
    num_anchors = all_anchors.shape[0]
    with tf.variable_scope('model_RPN') as scope:
        prediction_dict = RPN(feat_map, num_ref_anchors)
    
    # Get the tensors from the dict
    rpn_cls_prob = prediction_dict['rpn_cls_prob']
    rpn_bbox_pred = prediction_dict['rpn_bbox_pred']
    
    proposal_prediction = RPNProposal(rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape, nms_thresh)
    
    pred_dict_final['all_anchors'] = tf.cast(all_anchors, tf.float32)
    prediction_dict['proposals'] = proposal_prediction['proposals']
    prediction_dict['scores'] = proposal_prediction['scores']
        
    pred_dict_final['rpn_prediction'] = prediction_dict
    scores = pred_dict_final['rpn_prediction']['scores']
    proposals = pred_dict_final['rpn_prediction']['proposals']

    pred_masks_watershed = tf.to_float(marker_watershed(scores, proposals, pred_masks, min_score = bbox_min_score))
    

    # start point for testing, and end point for graph 

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_batches_test = len(x_test) 

    saver = tf.train.Saver()
    
    masks1 = []

    # Restore the per-image normalization model from the trained network
    saver.restore(sess,'./Network/whole_norm.ckpt')
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        # whole image normalization   
        batch_data = x_test[j]
        batch_data_shape = batch_data.shape
        image_normalized_wn = whole_image_norm(batch_data)
        image_normalized_wn = np.reshape(image_normalized_wn, [1,batch_data_shape[0],batch_data_shape[1],1])
        
        
        masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_wn})
        
        # First pass, get the coarse masks, and normalize the image on masks
        masks1.append(masks)
    
    # Restore the foreground normalization model from the trained network
    saver.restore(sess,'./Network/foreground.ckpt')
    #saver.restore(sess,'./Network/fg_norm_weights_fluorescent/'+str(30)+'.ckpt')
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        batch_data = x_test[j]
        batch_data_shape = batch_data.shape
        image = np.reshape(batch_data, [batch_data_shape[0],batch_data_shape[1]])
        
        # Final pass, foreground normalization to get final masks
        image_normalized_fg = foreground_norm(image, masks1[j])
        image_normalized_fg = np.reshape(image_normalized_fg, [1,batch_data_shape[0],batch_data_shape[1],1])
        
        # If adding watershed, we save the watershed masks separately
        if perform_watershed == 'yes':
            masks = sess.run(pred_masks_watershed, feed_dict={train_initial:image_normalized_fg})

            if postProcess == 'yes':
                masks = clean_image(masks)

        else:
            masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_fg})

            if postProcess == 'yes':
                masks = clean_image(masks)

        
    sess.close()
    
    return masks

def test_UNet(params, self):
    """Predict masks for all images in a given directory, and save them  

    Args:
        params (dict): the parameters of the network
    """

    postProcess = params['postProcess']
    resize_scale = params['scale_ratio']

    # Load the data
    # x_test, y_test: test images and corresponding labels
    x_id, x_test = load_data_test(self.batch_seg_path)
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (?,hf,wf,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(?,h,w,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)
    
    # start point for testing, and end point for graph 
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    num_batches_test = len(x_test) 

    saver = tf.train.Saver()
    # Restore the per-image normalization model from the trained network
    saver.restore(sess,'./Network/UNet.ckpt')
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        # whole image normalization
        batch_data = x_test[j]   
        batch_data_shape = batch_data.shape
        image = np.reshape(batch_data, [batch_data_shape[0],batch_data_shape[1]])

        if resize_scale != 1:
            image = rescale(image, self.params['scale_ratio'], anti_aliasing=True)

        # Clip the height and width to be 16-fold 
        imheight, imwidth = image.shape
        imheight = imheight//16*16
        imwidth = imwidth//16*16
        image = image[:imheight, :imwidth]

        image_normalized_wn = whole_image_norm(image)
        image_normalized_wn = np.reshape(image_normalized_wn, [1,imheight,imwidth,1])
        
        
        masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_wn})
        if not self.usingCL:
            self.progress_var.set(j/num_batches_test*100)
            self.window.update()
        if postProcess == 'yes':
            masks = clean_image(masks)

        # Revert the scale to original display
        if resize_scale != 1:
            masks = rescale(masks, 1/self.params['scale_ratio'])
            
        I8 = (((masks - masks.min()) / (masks.max() - masks.min())) * 255).astype(np.uint8)
        img = Image.fromarray(I8)
        img.save(self.batch_seg_path + x_id[j] + '_masks.png')
    sess.close()

# This function is similar to the function above, but only for one image that is 
# displayed on NuSeT GUI
def test_single_img_UNet(params, x_test):
    """input the image, return the segmented mask

    Args:
        params (dict): the parameters of the network
        x_test: the input image in numpy array
    """
    
    # Get the testing parameters 
    postProcess = params['postProcess']
    
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (?,32,32,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(?,512,512,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)

    # start point for testing, and end point for graph 

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_batches_test = len(x_test) 

    saver = tf.train.Saver()
    
    masks1 = []

    # Restore the per-image normalization model from the trained network
    saver.restore(sess,'./Network/UNet.ckpt')
    sess.run(tf.local_variables_initializer())
    for j in tqdm(range(0,num_batches_test)):
        # whole image normalization   
        batch_data = x_test[j]
        batch_data_shape = batch_data.shape
        image_normalized_wn = whole_image_norm(batch_data)
        image_normalized_wn = np.reshape(image_normalized_wn, [1,batch_data_shape[0],batch_data_shape[1],1])
        
        
        masks = sess.run(pred_masks, feed_dict={train_initial:image_normalized_wn})

        if postProcess == 'yes':
            masks = clean_image(masks)

    sess.close()
    
    return masks

