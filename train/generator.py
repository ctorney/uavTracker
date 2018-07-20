import os, sys, cv2
import copy
import numpy as np
from keras.utils import Sequence
sys.path.append("..")
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes, random_flip2, correct_bounding_boxes2

ANC_VALS = [[116,90],  [156,198],  [373,326],  [30,61], [62,45],  [59,119], [10,13],  [16,30],  [33,23]]

class BatchGenerator(Sequence):
    def __init__(self, 
        instances, 
        labels,        
        objects=1,        
        downsample=32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=30,
        batch_size=1,
        im_dir='./',
        min_net_size=320,
        max_net_size=608,    
        net_h=864,
        net_w=864,
        shuffle=True, 
        jitter=True 
    ):
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.objects            = objects
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.im_dir             = im_dir
 #       self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.net_h              = net_h
        self.net_w              = net_w

        if shuffle: np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))             # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, 3, 4+1+self.objects)) # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, 3, 4+1+self.objects)) # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, 3, 4+1+self.objects)) # desired network output 3
        yolos = [yolo_1, yolo_2, yolo_3]
        
        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)
            
            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None                
                max_index  = -1
                max_iou    = -1

                shifted_box = BoundBox(0, 0, obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'])    
                
                for i in range(len(ANC_VALS)):
                    anchor =BoundBox(0, 0, ANC_VALS[i][0],ANC_VALS[i][1]) 
                    iou    = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index  = i
                        max_iou    = iou                
                
                # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index//3]
                grid_h, grid_w = yolo.shape[1:3]
                
                # determine the position of the bounding box on the grid
                center_x = .5*(obj['xmin'] + obj['xmax'])
                g_center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
                center_y = .5*(obj['ymin'] + obj['ymax'])
                g_center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y
                
                # determine the sizes of the bounding box
                w = obj['xmax'] - obj['xmin']
                h = obj['ymax'] - obj['ymin']

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])  

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(g_center_x))
                grid_y = int(np.floor(g_center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
 #               yolo[instance_count, grid_y, grid_x, ]      = 0
                yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.
                yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1


            # assign input image to x_batch
            x_batch[instance_count] = img/255.

            # increase instance counter in the current batch
            instance_count += 1                 
                
  #      yolo_1 = yolo_1.reshape((yolo_1.shape[0],yolo_1.shape[1],yolo_1.shape[2],3*(self.objects+5)))
       # print(yolo_1.shape)
 #       return x_batch, yolo_3#  [dummy_yolo_1]
        return x_batch, [yolo_1, yolo_2, yolo_3]#  [dummy_yolo_1]
        return [x_batch, t_batch, yolo_1], [dummy_yolo_1]

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w
    
    def _aug_image(self, instance, net_h, net_w):
        _, image_name = os.path.split(instance['filename'])
        full_image_name = self.im_dir + image_name
        image = cv2.imread(full_image_name) # RGB image
        
        if image is None: print('Cannot find ', full_image_name)
        image = image[:,:,::-1] # RGB image
            
        image_h, image_w, _ = image.shape
        
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w;
        dh = self.jitter * image_h;

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
        scale = np.random.uniform(0.95, 1.05);

        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(net_w / new_ar);
            
        dx = int(np.random.uniform(0, net_w - new_w));
        dy = int(np.random.uniform(0, net_h - new_h));
        
        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
        
        # randomly distort hsv space
        # im_sized = random_distort_image(im_sized)
        
        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)
        #im_sized = random_flip(image, flip)
        #flip2 = np.random.randint(2)
        #im_sized = random_flip2(im_sized, flip2)
            
        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
        #all_objs = correct_bounding_boxes2(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, flip2, image_w, image_h)
        
        return im_sized, all_objs   

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])     
