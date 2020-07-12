import tensorflow as tf
from utils.tf_utils import activation_fun

def compile_model_FC(genome, nb_classes, inputs):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = genome.geneparam['nb_layers' ]
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]
    learning_rate = genome.geneparam['learning_rate' ]
    logging.info("Architecture:%s,%s,%s,%d,%f" % (nb_neurons[0:nb_layers], activation, optimizer, nb_layers, learning_rate))
    padding = 'same'
    # Add each layer.
    for i in range(0,nb_layers):
        # Need input shape for first layer.
        if i == 0:
            outputs = tf.layers.conv2d(inputs, nb_neurons[i], 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv'+str(i), use_bias=True)
            
        else:
            outputs = tf.layers.conv2d(outputs, nb_neurons[i], 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv'+str(i), use_bias=True)
        
        outputs = tf.layers.batch_normalization(outputs)
        outputs = activation_fun(outputs,activation)
        outputs = tf.nn.dropout(outputs,0.2)
    # only output 2 featuremap at the end
    outputs = tf.layers.conv2d(outputs, nb_classes, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final', use_bias=False)
    
    return outputs


def UNET(nb_classes, inputs):
    """Compile a UNET model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    #nb_layers  = genome.geneparam['nb_layers' ]
    #nb_neurons = genome.nb_neurons()

    padding = 'same'
    # Conv block 1
    outputs = tf.layers.conv2d(inputs, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1-1', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv1 output tensor 
    conv1_output = outputs
    
    # Down-sample 1
    outputs = tf.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Conv block 2
    outputs = tf.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv2 output tensor 
    conv2_output = outputs
    
    # Down-sample 2
    outputs = tf.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    # Now is ?,112,112,128
    
    # Conv block 3
    outputs = tf.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv3 output tensor 
    conv3_output = outputs
    
    # Down-sample 3
    outputs = tf.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Conv block 4
    outputs = tf.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv4 output tensor 
    conv4_output = outputs
    
    # Down-sample 4
    outputs = tf.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Get extracted feature for RPN
    rpn_feature = outputs
    
    
    # Conv block 5
    outputs = tf.layers.conv2d(outputs, 1024, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 1024, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    
    # Up-sample(Conv_transpose) 4
    outputs = tf.layers.conv2d_transpose(outputs, 512, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # changing the former line to the line below will connect the 4th layer on both ends, 
    # and form the classic U-Net architecture, but in our experiments we found this did 
    # not gain a better performance, also thanks for Meryem Uzun-Per for pointing this out
    
    # outputs = tf.concat([conv4_output, outputs], 3)

    
    # Conv block 4'
    outputs = tf.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 3
    outputs = tf.layers.conv2d_transpose(outputs, 256, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    outputs = tf.concat([conv3_output, outputs], 3)
    
    # Conv block 3'
    outputs = tf.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 2
    outputs = tf.layers.conv2d_transpose(outputs, 128, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    outputs = tf.concat([conv2_output, outputs], 3)
    
    # Conv block 2'
    outputs = tf.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 1
    outputs = tf.layers.conv2d_transpose(outputs, 64, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    outputs = tf.concat([conv1_output, outputs], 3)
    
    # Conv block 2'
    outputs = tf.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # only output 2 featuremaps at the end
    outputs = tf.layers.conv2d(outputs, nb_classes, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final', use_bias=False)
    
    return outputs, rpn_feature
