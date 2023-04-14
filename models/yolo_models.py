"""
    Yolo v3 implementation pre-trained on COCO.

    2019 Colin Torney

    Based on code from https://github.com/experiencor/keras-yolo3
    MIT License Copyright (c) 2017 Ngoc Anh Huynh

    Modified to process output within tensorflow and be agnostic to image size
"""
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Dense, Flatten, Activation, Reshape, Lambda, TimeDistributed, Permute, Conv3D
from tensorflow.keras.layers import Add, Concatenate
import tensorflow as tf

from tensorflow.keras import backend as K

ANC_VALS = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

def _conv_block(inp, convs, skip=True, train=False):
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
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True, trainable=trainflag)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']),trainable=trainflag)(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']),trainable=trainflag)(x)

    return Add()([skip_connection, x]) if skip else x

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

def positions():
    def func(z):
        x = z[0]
        y = z[1]
        # compute grid factor and net factor
        grid_h      = tf.shape(x)[1]
        grid_w      = tf.shape(x)[2]

        im_h      = tf.shape(y)[1]
        im_w      = tf.shape(y)[2]

        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
        net_factor  = tf.reshape(tf.cast([im_w, im_h], tf.float32), [1,1,1,1,2])

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(tf.maximum(grid_h,grid_w)), [tf.maximum(grid_h,grid_w)]), (1, tf.maximum(grid_h,grid_w), tf.maximum(grid_h,grid_w), 1, 1)),dtype=tf.float32)

        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, 3, 1])
        pred_box_xy = (cell_grid[:,:grid_h,:grid_w,:,:] + x)
        pred_box_xy = pred_box_xy * net_factor/grid_factor

        return pred_box_xy
    return Lambda(func)

def reshape_last_layer(out_size):
    def func(x):
        # reshape last 2 dimensions
        in_b      = tf.shape(x)[0]
        in_h      = tf.shape(x)[1]
        in_w      = tf.shape(x)[2]

        final_l = tf.reshape(x, [in_b, in_h, in_w, 3, out_size])
        return final_l

    return Lambda(func)

"""
Layers from 0 to 79 are the main detection block of Yolo
"""
def get_darknet_layers(input_image, trainable):

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}], train=trainable)

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}], train=trainable)

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}], train=trainable)

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}], train=trainable)

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}], train=trainable)

    skip_36 = x

    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}], train=trainable)

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}], train=trainable)

    skip_61 = x

    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}], train=trainable)

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}], train=trainable)

    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False, train=trainable)

    return x, skip_36, skip_61

def get_inner_layers(input_image, num_class, out_size, trainable, headtrainable):

    x, skip_36, skip_61 = get_darknet_layers(input_image, trainable)

    # Layer 80 => 82
    #x (75-79) creates an input for large_raw, which is layer 80
    yolo_80 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80}], skip=False, train=trainable)
    yolo_81 = _conv_block(yolo_80, [{'filter':  3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 'special_81'}], skip=False, train=trainable)
    yolo_82 = _conv_block(yolo_80, [{'filter':  3*num_class, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 'special_82'}], skip=False, train=trainable)

    # again using layer x (75-79) here
    # Layer 84 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False, train=trainable)
    x = UpSampling2D(2)(x) # layer 85
    x = Concatenate()([x, skip_61]) # layer 86

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False, train=trainable)

    # Layer 92 => 94

    # Here we are getting outputs for medium from x (87-91)
    yolo_92 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92}], skip=False, train=trainable)
    yolo_93 = _conv_block(yolo_92, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable, 'layer_idx': 'special_93'}], skip=False, train=trainable)
    yolo_94 = _conv_block(yolo_92, [{'filter': 3*num_class, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable, 'layer_idx': 'special_94'}], skip=False, train=trainable)

    # again using x conv(87-91)
    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False, train=trainable)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])

    # Layer 99 => 106
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},], skip=False, train=trainable)

    yolo_105 = x
    yolo_106 = _conv_block(yolo_105, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 106}], skip=False, train=trainable)
    yolo_107 = _conv_block(yolo_105, [{'filter': 3*num_class, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 107}], skip=False, train=trainable)

    inner_out_layers = [[yolo_81,yolo_82], [yolo_93,yolo_94], [yolo_106,yolo_107]]
    inner_out_layers_reshaped = []
    for fl in inner_out_layers:
        finashaped = reshape_last_layer(out_size)(fl[0])
        finashaped_class = reshape_last_layer(num_class)(fl[1])
        inner_out_layers_reshaped.append([finashaped,finashaped_class])


    raw_layers = [yolo_80, yolo_92, yolo_105]
    return inner_out_layers_reshaped, raw_layers

"""
This function is modification of Colin's code that provided
l_out = Concatenate()([l_offs, l_szs, l_obj, l_cls])
and instead provides
l_out = concatenate([l_offs, l_szs, l_scores, l_feats])
as in previous version of uavTracker
"""
def convert_output_layers(inner_out_layers, input_image, out_size, num_class):
    output = []
    anchor = 0

    for fl in inner_out_layers[:3]:

        finashaped = fl[0]
        finashaped_class = fl[1]
        # process centre points for grid offsets and convert to image coordinates
        offs = crop(0,2)(finashaped)
        offs = Activation('sigmoid')(offs)
        offs = positions()([offs, input_image])

        # process anchor boxes
        szs = crop(2,4)(finashaped)
        szs = anchors(anchor)(szs)
        anchor+=1

        # object confidence, aga scores
        obj = crop(4,5)(finashaped)
        obj = Activation('sigmoid')(obj)

        # class scores
        #cls = crop(out_size,out_size+num_class)(finashaped)
        cls = Activation('softmax')(finashaped_class)

        # combine results
        out = Concatenate()([offs, szs, cls, obj])
        output.append(out)

    if len(inner_out_layers) > 3:
        output = output + inner_out_layers[3:]

    return output

def get_layers(input_image, num_class, out_size, trainable, headtrainable, rawfeatures):
    # in previous versions we would have
    # large_raw = yolo_80
    # final_large = Reshape((in_h // 32, in_w // 32, 3, out_size))(yolo_81)
    # med_raw = yolo_92
    # final_med = Reshape((in_h // 16, in_w // 16, 3, out_size))(yolo_93)
    # small_raw = yolo_105
    # final_small = Reshape((in_h // 8, in_w // 8, 3, out_size))(yolo_106)

    # from get_layers to convert_output
    # return [final_large, final_med, final_small[, large_raw, med_raw, small_raw]

    # inner_out_layers = [[yolo_83,yolo_84], [yolo_93,yolo_94], [yolo_106,yolo_107]]
    inner_out_layers, raw_layers = get_inner_layers(input_image, num_class, out_size, trainable, headtrainable)

    if rawfeatures:
        inner_out_layers = inner_out_layers + raw_layers

    return inner_out_layers

def get_yolo_model(num_class=80,
                   trainable=False,
                   headtrainable=False,
                   rawfeatures=False):
    # for each box we have num_class outputs, 4 bbox coordinates, and 1 object confidence value
    out_size = num_class + 4 + 1 #out_size = 6
    input_image = Input(shape=(None, None, 3))
    inner_out_layers = get_layers(
            input_image, num_class, out_size, trainable,
            headtrainable, rawfeatures)

    output_layers = convert_output_layers(inner_out_layers, input_image, out_size, num_class)

    model = Model(input_image, output_layers)
    print(model.summary(line_length=120))

    return model


def get_train_base(weights_file,
                   num_class=1,
                   trainable=False):

    # for each box we have num_class outputs,
    # 4 bbox coordinates, and
    # 1 object confidence value
    # out_size = num_class + 8
    out_size = num_class + 4 + 1 # out_size = 6
    input_image = Input(shape=(None, None, 3))

    inner_out_layers = get_layers(
        input_image, num_class, out_size, trainable,
        headtrainable=False, rawfeatures=True)
    out_layers = [a[0] for a in inner_out_layers[:3]] + inner_out_layers[3:]
    detection_model = Model(input_image, out_layers)

    print('#############')
    print('Detection model')
    print('#############')
    print(detection_model.summary(line_length=120))

    detection_model.load_weights(weights_file, by_name=True)

    input_sequence = Input(shape=(3, None, None, 3))

    seq_large = TimeDistributed(
        Model(detection_model.input,
              detection_model.output[3]))(input_sequence)
    seq_med = TimeDistributed(
        Model(detection_model.input,
              detection_model.output[4]))(input_sequence)
    seq_small = TimeDistributed(
        Model(detection_model.input,
              detection_model.output[5]))(input_sequence)

    model = Model(input_sequence, [seq_large, seq_med, seq_small])

    print('#############')
    print('Train base model')
    print('#############')
    print(model.summary(line_length=150))
    return model

def rescale():
    def func(x):
        return (3.0 * x) - 1.0

    return Lambda(func)

def convert_tracker(final_large, final_med, final_small):

    s_offs = crop(0, 2)(final_small)
    s_szs = crop(2, 4)(final_small)
    s_szs = anchors(2)(s_szs)
    s_offs = Activation('sigmoid')(s_offs)
    s_offs = rescale()(s_offs)
    s_offs = positions()(s_offs)
    s_out = Concatenate()([s_offs, s_szs])

    m_offs = crop(0, 2)(final_med)
    m_szs = crop(2, 4)(final_med)
    m_szs = anchors(1)(m_szs)
    m_offs = Activation('sigmoid')(m_offs)
    m_offs = rescale()(m_offs)
    m_offs = positions()(m_offs)
    m_out = Concatenate()([m_offs, m_szs])

    l_offs = crop(0, 2)(final_large)
    l_szs = crop(2, 4)(final_large)
    l_szs = anchors(0)(l_szs)
    l_offs = Activation('sigmoid')(l_offs)
    l_offs = rescale()(l_offs)
    l_offs = positions()(l_offs)
    l_out = Concatenate()([l_offs, l_szs])

    return [l_out, m_out, s_out]

def get_tracker_model():

    in_large = Input(shape=(3, None, None, 1024))
    in_med = Input(shape=(3, None, None, 512))
    in_small = Input(shape=(3, None, None, 256))

    seq_large = Permute((2, 3, 1, 4))(in_large)
    seq_large = Conv3D(512, 3, padding='same')(seq_large)
    seq_large = BatchNormalization(epsilon=0.001,
                                   name='bnorm_seq_large')(seq_large)
    seq_large = LeakyReLU(alpha=0.1, name='leaky_seq_large')(seq_large)
    seq_large = Conv3D(4, 1, padding='same')(seq_large)

    seq_med = Permute((2, 3, 1, 4))(in_med)
    seq_med = Conv3D(512, 3, padding='same')(seq_med)
    seq_med = BatchNormalization(epsilon=0.001, name='bnorm_seq_med')(seq_med)
    seq_med = LeakyReLU(alpha=0.1, name='leaky_seq_med')(seq_med)
    seq_med = Conv3D(4, 1, padding='same')(seq_med)

    seq_small = Permute((2, 3, 1, 4))(in_small)
    seq_small = Conv3D(512, 3, padding='same')(seq_small)
    seq_small = BatchNormalization(epsilon=0.001,
                                   name='bnorm_seq_small')(seq_small)
    seq_small = LeakyReLU(alpha=0.1, name='leaky_seq_small')(seq_small)
    seq_small = Conv3D(4, 1, padding='same')(seq_small)

    outputs = convert_tracker(seq_large, seq_med, seq_small)

    model = Model([in_large, in_med, in_small], outputs)  #equence, raw_output)
    return model


def load_yolos(num_class, args_tracker, trained_detector_weights, trained_linker_weights):
    ##################################################
    print("Loading YOLO models")
    print("We will use the following model for testing of detection: ")
    print(trained_detector_weights)
    yolov3 = get_yolo_model(num_class, trainable=False, rawfeatures = args_tracker)
    yolov3.load_weights(
        trained_detector_weights, by_name=False)
    print("We will use the following model for testing of linking: ")
    print(trained_linker_weights)
    yolov3link = get_tracker_model()
    yolov3link.load_weights(
        trained_linker_weights, by_name=False)
    print("YOLO models loaded, my dear.")
    ########################################
    return yolov3, yolov3link

class yolo_model():

    def __init__(self, labelclassMap, state, alltrain=False):

        self.labelclassMap = labelclassMap
        self.numClasses = len(labelclassMap.keys())
        self.alltrain = alltrain
        self.yolo_nn = get_yolo_model(self.numClasses, trainable=self.alltrain, headtrainable=True)

        timestampStr = datetime.now().strftime("%Y%m%d%H%M%S")

        self.state = (state if state is not None else 'weights/' + timestampStr)




    def getStateDict(self):
        if self.state is not None:
            self.yolo_nn.save_weights(self.state + '.h5')
            self.yolo_nn.save(self.state + '_full.h5')

        stateDict = {
            'model_state': self.state,
            'labelclassMap': self.labelclassMap,
            'alltrain': self.alltrain
        }
        return stateDict


    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        alltrain = (stateDict['alltrain'] if 'alltrain' in stateDict else False)
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)
        init_weights = (stateDict['init_weights'] if 'init_weights' in stateDict else None)

        # return model
        model = yolo_model(labelclassMap, state)
        if state is not None:
            print('loading saved weights ', state)
            model.yolo_nn.load_weights(state + '.h5')
        elif init_weights is not None:
            print('loading initial weights')
            weights_file = tf.keras.utils.get_file(origin=init_weights)
                    #"https://www.dropbox.com/s/3ra7a829w1f9hkl/mara-yolo.h5?dl=1")

            model.yolo_nn.load_weights(weights_file)
        return model
