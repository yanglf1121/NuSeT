import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

from model_layers.models import UNET
from model_layers.model_RPN import RPN
from model_layers.anchor_size import anchor_size
from model_layers.rpn_target import RPNTarget
from model_layers.rpn_proposal import RPNProposal
from model_layers.rpn_loss import RPNLoss
from model_layers.seg_loss import segmentation_loss
from model_layers.marker_watershed import marker_watershed
from model_layers.compute_metrics import compute_metrics

from utils.load_data import load_data_train
from utils.tf_utils import optimizer_fun
from utils.anchors import generate_anchors_reference
from utils.generate_anchors import generate_anchors
from utils.test import generate_gt_boxes
from utils.normalization import whole_image_norm, foreground_norm
from utils.losses import smooth_l1_loss
from utils.image_vis import draw_rpn_bbox_pred, draw_gt_boxes, draw_top_nms_proposals, draw_rpn_bbox_targets, draw_rpn_bbox_pred_only 


# inspired from https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/rpn_test.py
def train_NuSeT(self):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
    return:
        accuracy (float)
    """
    # Get the training parameters    
    learning_rate = self.params['lr']
    optimizer = self.params['optimizer'] 
    num_epoch = self.params['epochs']
    bbox_min_score = self.params['min_score'] 
    nms_thresh = self.params['nms_threshold']
    normalization_method = self.params['normalization_method']
    # Load the data
    # x_train, y_train: training images and corresponding labels
    # x_val, y_val: validation images and corresponding labels
    # w_train, w_val: training and validation weight matrices for U-Net
    # bbox_train, bbox_val: bounding box coordinates for train and validation dataset
    x_train, x_val, y_train, y_val, w_train, w_val, bbox_train, bbox_val = load_data_train(self, normalization_method)
    
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    # tensor placeholder for training images with labels
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    labels = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    
    # tensor placeholder for weigth matrices and ground truth bounding boxes
    edge_weights = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    gt_boxes = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    
    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (1,input_height/16,input_width/16,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(1,input_height,input_width,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)
    
    # Dynamic anchor base size calculated from median cell lengths
    base_size = anchor_size(tf.reshape(labels,[input_height,input_width]))

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
    pred_dict_final['gt_bboxes'] = gt_boxes
    prediction_dict['proposals'] = proposal_prediction['proposals']
    prediction_dict['scores'] = proposal_prediction['scores']
    
    # When training we use a separate module to calculate the target
    # values we want to output.
    (rpn_cls_target, rpn_bbox_target,rpn_max_overlap) = RPNTarget(all_anchors, num_anchors, gt_boxes, im_shape)

    prediction_dict['rpn_cls_target'] = rpn_cls_target
    prediction_dict['rpn_bbox_target'] = rpn_bbox_target
        
    
    pred_dict_final['rpn_prediction'] = prediction_dict
    scores = pred_dict_final['rpn_prediction']['scores']
    proposals = pred_dict_final['rpn_prediction']['proposals']
    
    pred_masks_watershed = tf.to_float(marker_watershed(scores, proposals, pred_masks, min_score=bbox_min_score))
    
    # Loss is defined as rpn loss(class loss + bounding box loss) + 
    # segmentation loss(default is the sum of soft dice and cross-entropy)
    rpn_loss = RPNLoss(prediction_dict)    

    RPN_loss = rpn_loss['rpn_cls_loss'] + rpn_loss['rpn_reg_loss']
    SEG_loss = segmentation_loss(final_logits, pred_masks_watershed, labels, edge_weights, mode = 'COMBO')
    
    final_loss = RPN_loss + SEG_loss
    
    # If training with just U-Net, then only include segmentation loss
    #final_loss = SEG_loss
    
    # Metrics are pixel accuracy, mean IU, mean accuracy, root mean squared error
    metrics, metrics_op = compute_metrics(pred_masks, labels)
    
    pred_dict_final['unet_mask'] = pred_masks 

    # get the optimizer
    gen_train_op = optimizer_fun(optimizer, final_loss, learning_rate=learning_rate)

    # start point for training, and end point for graph 
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())

    num_batches = len(x_train)
    num_batches_val = len(x_val) 

    saver = tf.train.Saver()

    if normalization_method == 'wn':
        self.training_results.set('Start whole image Norm. training ...')
        self.window.update()
    
    if normalization_method == 'fg':
        self.training_results.set('Start Foreground Norm. training ...')
        self.window.update()

    # training images indexes will be shuffled at every epoch during training
    idx = np.arange(num_batches)
    
    best_IU = 0
    for iteration in range(0,num_epoch):
        # The batch pointer to validation data
        j = 0
        sess.run(tf.local_variables_initializer())
        if iteration == num_epoch - 1 and normalization_method == 'wn':
            self.whole_norm_y_pred = []

        # shuffle the sequence of the training data for the current epoch
        np.random.shuffle(idx)

        for i in tqdm(range(0,num_batches)):
            self.train_progress_var.set(i/num_batches*100)
            self.window.update()
            # Generate the batch data from training data and training label
            batch_data = x_train[idx[i]]
            batch_data_shape = batch_data.shape            
            batch_data = np.reshape(batch_data, [1,batch_data_shape[0],batch_data_shape[1],1])
            batch_label = np.reshape(y_train[idx[i]], [1,batch_data_shape[0],batch_data_shape[1],1])
            batch_edge = np.reshape(w_train[idx[i]], [1,batch_data_shape[0],batch_data_shape[1],1])
            batch_bbox = bbox_train[idx[i]]
            
            # Skip if this batch does not contain any object (bounding box is null)
            if batch_bbox.size > 0:
                # Here include the optimizer to actually perform learning
                sess.run([gen_train_op], feed_dict={train_initial:batch_data, gt_boxes:batch_bbox, labels:batch_label, edge_weights:batch_edge})

                # Only calculate the accuracy and loss after the training epoch
                if i == num_batches - 1:
                    while j < num_batches_val:
                        # Generate the batch data from val data and val label

                        batch_data = x_val[j]
                        batch_data_shape = batch_data.shape  
                        batch_data = np.reshape(batch_data, [1,batch_data_shape[0],batch_data_shape[1],1])
                        batch_label = np.reshape(y_val[j], [1,batch_data_shape[0],batch_data_shape[1],1])
                        batch_edge = np.reshape(w_val[j], [1,batch_data_shape[0],batch_data_shape[1],1])
                        batch_bbox = bbox_val[j]
                        
                        # At the end of whole image normalization training,
                        # cache the predictions 
                        if iteration == num_epoch - 1 and normalization_method == 'wn':
                            self.whole_norm_y_pred.append(sess.run(pred_masks, 
                            feed_dict={train_initial:batch_data, gt_boxes:batch_bbox, labels:batch_label, edge_weights:batch_edge}))


                        if batch_bbox.size > 0:
                            # Here get the accuracy and loss for each batch in validation cycle
                            loss_temp, rpnloss_temp, segloss_temp  = sess.run([final_loss, rpn_loss, SEG_loss], feed_dict={train_initial:batch_data, gt_boxes:batch_bbox, labels:batch_label, edge_weights:batch_edge})

                            sess.run([metrics_op], feed_dict={train_initial:batch_data, gt_boxes:batch_bbox, labels:batch_label, edge_weights:batch_edge})
                            
                            if j == num_batches_val - 1:
                                metrics_all  = sess.run(metrics, feed_dict={train_initial:batch_data, gt_boxes:batch_bbox, labels:batch_label, edge_weights:batch_edge})
                            
                                _mean_IU = metrics_all['global']['mean_IU']
                                _pixel_accuracy = metrics_all['global']['pixel_accuracy']
                                _f1 = 2 * _mean_IU / (1 + _mean_IU)
                                _rmse = metrics_all['global']['rmse']
                                
                            # Get moving average of metrics and losses
                            if j == 0:
                                loss_total = loss_temp
                                cls_loss = rpnloss_temp['rpn_cls_loss']
                                reg_loss = rpnloss_temp['rpn_reg_loss']
                                seg_loss = segloss_temp

                            else:
                                loss_total = (1 - 1 / (j + 1)) * loss_total + 1 / (j + 1) * loss_temp
                                cls_loss = (1 - 1 / (j + 1)) * cls_loss + 1 / (j + 1) * rpnloss_temp['rpn_cls_loss']
                                reg_loss = (1 - 1 / (j + 1)) * reg_loss + 1 / (j + 1) * rpnloss_temp['rpn_reg_loss']
                                seg_loss = (1 - 1 / (j + 1)) * seg_loss + 1 / (j + 1) * segloss_temp
                            
                            
                            j = j + 1

        print('Epoch: %d - loss: %.2f - cls_loss: %.2f - reg_loss: %.2f - seg_loss: %.2f - mean_IU: %.4f - f1: %.4f - pixel_accuracy: %.4f' % (iteration, loss_total, cls_loss, reg_loss, seg_loss, _mean_IU, _f1, _pixel_accuracy))

        self.training_results.set('Epoch' + str(iteration) +
 ', loss ' + '{0:.2f}'.format(loss_total) + ', mean IU ' + '{0:.2f}'.format(_mean_IU))
        self.window.update()
	
        # Keep track of the best model in the last 10 epoches and use that as the best model
        
        if iteration >= num_epoch - 10 and normalization_method == 'wn' and _mean_IU > best_IU:
            best_IU = _mean_IU
            saver.save(sess, './Network/whole_norm.ckpt')

        if iteration >= num_epoch - 10 and normalization_method == 'fg' and _mean_IU > best_IU:
            best_IU = _mean_IU
            saver.save(sess, './Network/foreground.ckpt')
        
    sess.close()

# Train the pure U-Net model
def train_UNet(self):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
    return:
        accuracy (float)
    """
    # Get the training parameters    
    learning_rate = self.params['lr']
    optimizer = self.params['optimizer'] 
    num_epoch = self.params['epochs']
    normalization_method = self.params['normalization_method']
    # Load the data
    # x_train, y_train: training images and corresponding labels
    # x_val, y_val: validation images and corresponding labels
    # w_train, w_val: training and validation weight matrices for U-Net
    # bbox_train, bbox_val: bounding box coordinates for train and validation dataset
    x_train, x_val, y_train, y_val, w_train, w_val, bbox_train, bbox_val = load_data_train(self, 'wn')
    
    # pred_dict and pred_dict_final save all the temp variables
    pred_dict_final = {}
    
    # tensor placeholder for training images with labels
    train_initial = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    labels = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    
    # tensor placeholder for weigth matrices and ground truth bounding boxes
    edge_weights = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    
    input_shape = tf.shape(train_initial)
    
    input_height = input_shape[1]
    input_width = input_shape[2]
    im_shape = tf.cast([input_height, input_width], tf.float32)
    
    # number of classes needed to be classified, for our case this equals to 2
    # (foreground and background)
    nb_classes = 2
    
    
    # feed the initial image to U-Net, we expect 2 outputs: 
    # 1. feat_map of shape (1,input_height/16,input_width/16,1024), which will be passed to the 
    # region proposal network
    # 2. final_logits of shape(1,input_height,input_width,2), which is the prediction from U-net
    with tf.variable_scope('model_U-Net') as scope:
        final_logits, feat_map = UNET(nb_classes, train_initial)
    
    # The final_logits has 2 channels for foreground/background softmax scores,
    # then we get prediction with larger score for each pixel
    pred_masks = tf.argmax(final_logits, axis=3)
    pred_masks = tf.reshape(pred_masks,[input_height,input_width])
    pred_masks = tf.to_float(pred_masks)
    
    SEG_loss = segmentation_loss(final_logits, pred_masks, labels, edge_weights, mode = 'COMBO')
    
    final_loss = SEG_loss
    
    # Metrics are pixel accuracy, mean IU, mean accuracy, root mean squared error
    metrics, metrics_op = compute_metrics(pred_masks, labels)
    
    pred_dict_final['unet_mask'] = pred_masks 

    # get the optimizer
    gen_train_op = optimizer_fun(optimizer, final_loss, learning_rate=learning_rate)

    # start point for training, and end point for graph 
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())

    num_batches = len(x_train)
    num_batches_val = len(x_val) 

    saver = tf.train.Saver()

    self.training_results.set('U-Net: Start training ...')
    self.window.update()

    # training images indexes will be shuffled at every epoch during training
    idx = np.arange(num_batches)
    
    best_IU = 0
    for iteration in range(0,num_epoch):
        # The batch pointer to validation data
        j = 0
        sess.run(tf.local_variables_initializer())
        if iteration == num_epoch - 1 and normalization_method == 'wn':
            self.whole_norm_y_pred = []

        # shuffle the sequence of the training data for the current epoch
        np.random.shuffle(idx)

        for i in tqdm(range(0,num_batches)):
            self.train_progress_var.set(i/num_batches*100)
            self.window.update()
            # Generate the batch data from training data and training label
            batch_data = x_train[idx[i]]
            batch_data_shape = batch_data.shape            
            batch_data = np.reshape(batch_data, [1,batch_data_shape[0],batch_data_shape[1],1])
            batch_label = np.reshape(y_train[idx[i]], [1,batch_data_shape[0],batch_data_shape[1],1])
            batch_edge = np.reshape(w_train[idx[i]], [1,batch_data_shape[0],batch_data_shape[1],1])

            # Here include the optimizer to actually perform learning
            sess.run([gen_train_op], feed_dict={train_initial:batch_data, labels:batch_label, edge_weights:batch_edge})

            # Only calculate the accuracy and loss after the training epoch
            if i == num_batches - 1:
                while j < num_batches_val:
                    # Generate the batch data from val data and val label

                    batch_data = x_val[j]
                    batch_data_shape = batch_data.shape  
                    batch_data = np.reshape(batch_data, [1,batch_data_shape[0],batch_data_shape[1],1])
                    batch_label = np.reshape(y_val[j], [1,batch_data_shape[0],batch_data_shape[1],1])
                    batch_edge = np.reshape(w_val[j], [1,batch_data_shape[0],batch_data_shape[1],1])

                    loss_temp  = sess.run(final_loss, feed_dict={train_initial:batch_data, labels:batch_label, edge_weights:batch_edge})

                    sess.run([metrics_op], feed_dict={train_initial:batch_data, labels:batch_label, edge_weights:batch_edge})
                    
                    if j == num_batches_val - 1:
                        metrics_all  = sess.run(metrics, feed_dict={train_initial:batch_data, labels:batch_label, edge_weights:batch_edge})
                    
                        _mean_IU = metrics_all['global']['mean_IU']
                        _pixel_accuracy = metrics_all['global']['pixel_accuracy']
                        _f1 = 2 * _mean_IU / (1 + _mean_IU)
                        _rmse = metrics_all['global']['rmse']
                        
                    # Get moving average of metrics and losses
                    if j == 0:
                        loss_total = loss_temp
                    else:
                        loss_total = (1 - 1 / (j + 1)) * loss_total + 1 / (j + 1) * loss_temp
                                    
                    j = j + 1

        print('Epoch: %d - loss: %.2f - mean_IU: %.4f - f1: %.4f - pixel_accuracy: %.4f' % (iteration, loss_total, _mean_IU, _f1, _pixel_accuracy))

        self.training_results.set('Epoch' + str(iteration) +
', loss ' + '{0:.2f}'.format(loss_total) + ', mean IU ' + '{0:.2f}'.format(_mean_IU))
        self.window.update()

    # Keep track of the best model in the last 10 epoches and use that as the best model
    
        if iteration >= num_epoch - 10 and normalization_method == 'wn' and _mean_IU > best_IU:
            best_IU = _mean_IU
            saver.save(sess, './Network/UNet.ckpt')
        
    sess.close()
    

