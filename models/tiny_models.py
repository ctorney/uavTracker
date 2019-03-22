
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Dense, Flatten, Activation, Reshape, Lambda
from keras.layers.merge import add, concatenate
import tensorflow as tf

from keras import backend as K

ANC_VALS = [[81,82, 135,169, 344,319],[10,14, 23,27, 37,58 ]]

def _conv_block(inp, convs, skip=False, train=False):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
        if 'train' in conv:
            trainflag=conv['train']#update the value for the key
        else:
            trainflag=train
        x = Conv2D(conv['filter'], 
                   conv['kernel'], 
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv2d_' + str(conv['layer_idx']), 
                   use_bias=False if conv['bnorm'] else True, trainable=trainflag)(x)
        #if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='batch_normalization' + str(conv['layer_idx']),trainable=trainflag)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, trainable=trainflag)(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']),trainable=trainflag)(x)

    return add([skip_connection, x]) if skip else x

def crop(start, end):
    # Crops (or slices) a Tensor on fourth dimension from start to end
    def func(x):
        return x[:, :, :, :, start: end]
    return Lambda(func)

def anchors(i):
    def func(x):
        anc = tf.constant(ANC_VALS[i], dtype='float', shape=[1,1,1,3,2])
        return tf.exp(x) * anc 
    return Lambda(func)

def positions(h,w):
    def func(x):
        # compute grid factor and net factor
        grid_h      = tf.shape(x)[1]
        grid_w      = tf.shape(x)[2]

        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
        net_factor  = tf.reshape(tf.cast([w, h], tf.float32), [1,1,1,1,2])
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(tf.maximum(grid_h,grid_w)), [tf.maximum(grid_h,grid_w)]), (1, tf.maximum(grid_h,grid_w), tf.maximum(grid_h,grid_w), 1, 1)))

        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, 3, 1])
        pred_box_xy = (cell_grid[:,:grid_h,:grid_w,:,:] + x) 
        pred_box_xy = pred_box_xy * net_factor/grid_factor 

        return pred_box_xy 
    return Lambda(func)

def get_tiny_model(in_w=416,in_h=416, num_class=80, trainable=False, headtrainable=False):

    # for each box we have num_class outputs, 4 bbox coordinates, and 1 object confidence value
    out_size = num_class+5
    input_image = Input(shape=( in_h,in_w, 3))

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 16, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0}], train=trainable)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    x = _conv_block(x, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2}], train=trainable)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    x = _conv_block(x, [{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 4}], train=trainable)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6}], train=trainable)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 8}], train=trainable)
    skip_x1 = x    
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}], train=trainable)
    x = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x)
    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13}], train=trainable)

    skip_layer = x
    # Layer 80 => 82
    if num_class!=80:
        yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 14},
                              {'filter':  3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 915}], skip=False, train=trainable)
    else:
        yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 14},
                              {'filter':  3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 15}], skip=False, train=trainable)

    
    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 18}], skip=False, train=trainable)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_x1])

    # Layer 99 => 106
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 21},], skip=False, train=trainable)

    if num_class!=80:
        yolo_106 = _conv_block(x, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 922}], skip=False, train=trainable)
    else:
        yolo_106 = _conv_block(x, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 22}], skip=False, train=trainable)


 #   final_large = Reshape((in_h//32,in_w//32,3,out_size))(yolo_82)
    final_med = Reshape((in_h//32, in_w//32,3,out_size))(yolo_94)
    final_small = Reshape((in_h//16,in_w//16,3,out_size))(yolo_106)
 #   output = [yolo_94, yolo_106]  
 #   model = Model(input_image,output)
#    return model

    s_offs =crop(0,2)(final_small)
    s_szs =crop(2,4)(final_small)
    s_scores =crop(4,out_size)(final_small)
    s_scores = Activation('sigmoid')(s_scores)
    s_szs = anchors(1)(s_szs)
    s_offs = Activation('sigmoid')(s_offs)
    s_offs = positions(in_h,in_w)(s_offs)
    s_out = concatenate([s_offs, s_szs, s_scores])

    m_offs =crop(0,2)(final_med)
    m_szs =crop(2,4)(final_med)
    m_scores =crop(4,out_size)(final_med)
    m_scores = Activation('sigmoid')(m_scores)
    m_szs = anchors(0)(m_szs)
    m_offs = Activation('sigmoid')(m_offs)
    m_offs = positions(in_h,in_w)(m_offs)
    m_out = concatenate([m_offs, m_szs, m_scores])

 #   l_offs =crop(0,2)(final_large)
 #   l_szs =crop(2,4)(final_large)
 #   l_scores =crop(4,out_size)(final_large)
 #   l_scores = Activation('sigmoid')(l_scores)
 #   l_szs = anchors(0)(l_szs)
 #   l_offs = Activation('sigmoid')(l_offs)
 #   l_offs = positions(in_h,in_w)(l_offs)
 #   l_out = concatenate([l_offs, l_szs, l_scores])

    output = [m_out, s_out]  

    model = Model(input_image,output)
    return model





