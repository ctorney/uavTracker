"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import sys
import os.path
import scipy
import numpy as np
sys.path.append('..')
from utils.linear_assignment import linear_assignment
from scipy.optimize import linear_sum_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from filterpy.stats import mahalanobis
import filterpy
from utils.decoder import do_nms, bbox_iou, interval_overlap

def convert_bbox_to_kfx(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    return np.array([x,y,w,h]).reshape((4,1))

def convert_kfx_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w=max(0.0,x[2])
    h=max(0.0,x[3])
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def get_box(trk):
    if trk.kalman_type == 'torney':
        trackBox = convert_kfx_to_bbox(trk.kf.x[:4])[0]
    elif trk.kalman_type == 'sort':
        trackBox = convert_x_to_bbox(trk.kf.x[:4])[0]
    elif trk.kalman_type == 'deepbeast':
        trackBox = trk.bbox[:4]
    else:
        raise ValueError('Unknown kalman type')
    return trackBox 

class KalmanBoxSortTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  kalman_type='sort'
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxSortTracker.count
    KalmanBoxSortTracker.count += 1
    self.hits = 0
    self.hit_streak = 0
    self.age = 1
    self.score = bbox[4]
    self.long_score = bbox[4]/2 #when creating track the long_score is halved to indicate our uncertainty over one high-confidence detection

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    self.score = (self.score*(self.hits-1.0)/float(self.hits)) + (bbox[4]/float(self.hits))
    self.long_score = (self.long_score*(self.age-1.0)/float(self.age)) + (bbox[4]/float(self.age)) #average of the entire track
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.long_score = (self.long_score*(self.age-1.0)/float(self.age)) #We are back-filling the score for the missed detection from the previous update
    self.time_since_update += 1

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    kalman_type='torney'
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0.5,0],[0,1,0,0,0,1,0,0.5],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]])

        self.kf.R[:,:] *= 25.0 # set measurement uncertainty for positions
        self.kf.Q[:2,:2] = 0.0 # process uncertainty for positions is zero - only moves due to velocity, leave process for width height as 1 to account for turning
        self.kf.Q[2:4,2:4] *= 5 # process uncertainty for width/height for turning
        self.kf.Q[4:6,4:6] = 0.0 # process uncertainty for velocities is zeros - only accelerates due to accelerations
        self.kf.Q[6:,6:] *= 0.01 # process uncertainty for acceleration
        self.kf.P[4:,4:] *= 15.0 # maximum speed

        z=convert_bbox_to_kfx(bbox)
        self.kf.x[:4] = z
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.score = bbox[4]
        self.long_score = bbox[4]/2 #when creating track the long_score is halved to indicate our uncertainty over one high-confidence detection

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = (self.score*(self.hits-1.0)/float(self.hits)) + (bbox[4]/float(self.hits)) #average of the entire track
        self.long_score = (self.long_score*(self.age-1.0)/float(self.age)) + (bbox[4]/float(self.age)) #average of the entire track
        z = convert_bbox_to_kfx(bbox)
        self.kf.update(z)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.long_score = (self.long_score*(self.age-1.0)/float(self.age)) #We are back-filling the score for the missed detection from the previous update

        self.time_since_update += 1

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_kfx_to_bbox(self.kf.x)

    def get_distance(self, y):
        """
        Returns the mahalanobis distance to the given point.
        """
        b1 = convert_kfx_to_bbox(self.kf.x[:4])[0]
        return (bbox_iou(b1,y))

class BeastTrack(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        self.bbox = bbox
        self.time_since_update = 0
        self.id = BeastTrack.count
        BeastTrack.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.score = bbox[4]
        self.long_score = bbox[4]/2 #when creating track the long_score is halved to indicate our uncertainty over one high-confidence detection
        self.kalman_type='deepbeast'

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = (self.score*(self.hits-1.0)/float(self.hits)) + (bbox[4]/float(self.hits))
        self.long_score = (self.long_score*(self.age-1.0)/float(self.age)) + (bbox[4]/float(self.age)) #average of the entire track
        self.bbox = bbox

    def predict_fill(self, bbox):
        self.bbox = bbox

    def pre_predict(self):
        '''
        This is an equivalent of Kalman Filter predict function. However as the predictions happen on the global level of the battery, we are passing a new predicted bbox from battery predictions in predict_fill
        '''
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.long_score = (self.long_score*(self.age-1.0)/float(self.age))
        
        self.time_since_update += 1

'''
DeepBeastTrackerBettery
Unlike Kalman Box Trackers, for DeepBeastTrackerBattery we are only having one tracker for all tracks.
'''
class DeepBeastTrackerBattery(object):
    def __init__(self, yolo_det_model, yololink, obj_thresh, nms_thresh) -> None:
       self.yolo_det_model = yolo_det_model
       self.yololink = yololink 
       self.trackers = []
       self.obj_thresh = obj_thresh
       self.nms_thresh = nms_thresh
       #Initialise the list of tracker objects

    def create_trackers(self, detections):
        '''
        Create a list of tracker objects from the detections.
        '''
        for det in detections:
            self.trackers.append(BeastTrack(det))      
        return self.trackers

    def decodeLinker(self, track_output,current_tracks):
        new_boxes = np.empty((0,9))

        for trk in current_tracks:
            #Get yolo anchor box for prevoius frame detection 
            c0 = int(trk.bbox[5])
            c1 = int(trk.bbox[6])
            c2 = int(trk.bbox[7])
            #index of previous detection
            ipd = (c0, c1, c2)
            yolo_scale = int(trk.bbox[8])

            if c0 == -1: #If for some reason the track in previous frame was not linked to any detection
                n_box = trk.bbox[:4]
            else:
                trakcpred = track_output[yolo_scale][0]
                xpos = trakcpred[...,0]
                ypos = trakcpred[...,1]
                wpos = trakcpred[...,2]
                hpos = trakcpred[...,3]

                n_box = [xpos[ipd]-wpos[ipd]/2.0, 
                        ypos[ipd]-hpos[ipd]/2.0, 
                        xpos[ipd]+wpos[ipd]/2.0, 
                        ypos[ipd]+hpos[ipd]/2.0, 
                        0,
                        c0,
                        c1,
                        c2,
                        yolo_scale] #we are setting the score to 0 here as we are not suure if there will be later detection. Also our best guess at location of the next prediction would be previous yolo location in absence of detection
            new_boxes = np.vstack((new_boxes,n_box))

        return new_boxes

    def predict(self, frame_a, frame_b, frame_c):
        current_linker_tracks = self.trackers

        new_image_a = frame_a[:, :, ::-1] / 255.#opencvuses BGR,. rest of the world RGB (inc yolo)
        new_image_a = np.expand_dims(new_image_a, 0)
        new_image_b = frame_b[:, :, ::-1] / 255.#opencvuses BGR,. rest of the world RGB (inc yolo)
        new_image_b = np.expand_dims(new_image_b, 0)
        new_image_c = frame_c[:, :, ::-1] / 255.#opencvuses BGR,. rest of the world RGB (inc yolo)
        new_image_c = np.expand_dims(new_image_c, 0)

        # run the prediction
        sys.stdout.write('Yolo predicting in linker ...')
        sys.stdout.flush()
        yolos_a = self.yolo_det_model.predict(new_image_a)
        yolos_b = self.yolo_det_model.predict(new_image_b)
        yolos_c = self.yolo_det_model.predict(new_image_c)

        large_seq = np.concatenate((yolos_a[3],yolos_b[3],yolos_c[3]))
        med_seq = np.concatenate((yolos_a[4],yolos_b[4],yolos_c[4]))
        small_seq = np.concatenate((yolos_a[5],yolos_b[5],yolos_c[5]))

        track_seq = [large_seq[None,...],med_seq[None,...],small_seq[None,...]]
        track_output = self.yololink.predict(track_seq)

        #get The box from the linker/tracker
        c_linker_boxes = []
        #decodeLinker returns matching boxes from frame C and B (t and t-1)
        #The linker decoder should take the ground truth position at t-1 (B), or boxes_gt_prev and form this provide us with its best estimate of boxes_gt at t
        new_boxes = self.decodeLinker(track_output, current_linker_tracks)
        return new_boxes

    def predict_trackers(self, frame_a, frame_b, frame_c):
        '''
        For all of the trackers in this battery and predict their next position based on deep beast linker. Here we are not yet associating the trackers with the detections, we simply provide a prediction for each *previously existing* tracker
        '''
        #Get actual predictions of positions of all objects in frame b using deepbeast
        c_pred_boxes = self.predict(frame_a, frame_b, frame_c)

        #Do the technical update of all trackers state
        for iii, trk in enumerate(self.trackers):
            #For each tracker, assign correct c_pred_box as the predicted location
            trk.pre_predict()
            trk.predict_fill(c_pred_boxes[iii])
        
        return self.trackers

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    id_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    scale_id = 0.5

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            trackBox = get_box(trk)
            iou_matrix[d,t] = bbox_iou(trackBox, det)
            id_matrix[d,t] = scale_id*det[4]
    matched_indices = linear_assignment(-iou_matrix-id_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low probability
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class yoloTracker(object):
    def __init__(self,max_age=1,track_threshold=0.5, init_threshold=0.9, init_nms=0.0,link_iou=0.3, hold_without=3, kalman_type='torney', yolo_det_model = None, yololink = None ):
        """
        Sets key parameters for YOLOtrack
        """
        self.hold_without = hold_without
        self.max_age = max_age # time since last detection to delete track
        self.trackers = []
        self.frame_count = 0
        self.track_threshold = track_threshold # only return tracks with average confidence above this value
        self.init_threshold = init_threshold # threshold confidence to initialise a track, note this is much higher than the detection threshold
        self.init_nms = init_nms # threshold overlap to initialise a track - set to 0 to only initialise if not overlapping another tracked detection
        self.link_iou = link_iou # only link tracks if the predicted box overlaps detection by this amount
        self.kalman_type = kalman_type
        if self.kalman_type == 'sort':
            KalmanBoxSortTracker.count = 0
        elif self.kalman_type == 'torney':
            KalmanBoxTracker.count = 0
        elif self.kalman_type == 'deepbeast':
            self.dbt_battery = DeepBeastTrackerBattery(yolo_det_model, yololink, track_threshold, init_nms)
        else:
            raise Exception('Unknown name of a tracker: {self.kalman_type}')

    def update_0_predict(self, frame_a=None, frame_b=None, frame_c=None):
        '''
        Update all trackers with new detections predictions
        '''
        self.frame_count += 1
        
        #get predicted locations from existing trackers for Kalman
        if self.kalman_type != 'deepbeast':
            for t,trk in enumerate(self.trackers):
                self.trackers[t].predict()

        #Get all predictions from Deep Beast Battery
        else:
            #we are updating yoloTracker trackers wtih DeepBeastTrackerBattery trackers
            self.trackers = self.dbt_battery.predict_trackers(frame_a, frame_b, frame_c)

        ret = []
        for trk in (self.trackers):
            d = get_box(trk)
            ret.append(np.concatenate((d,[trk.id,trk.long_score,trk.score])).reshape(1,-1))

        #The following lines are very important because python secretly cares about data types very much.
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

    def update_1_update(self, dets):
        ret = []
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,self.trackers, self.link_iou)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0])
                dets[d,4]=2.0 # once assigned we set it to full certainty

        #add umatched tracks to detection list
        for t,trk in enumerate(self.trackers):
            if(t in unmatched_trks):
                d = get_box(trk)
                #yolo grid indeces are set to -1
                d = np.append(d,np.array([2,-1,-1,-1,-1]), axis=0)
                d = np.expand_dims(d,0)

                if len(dets)>0:
                    dets = np.append(dets,d, axis=0)
                else:
                    dets = d

        if len(dets)>0:
            dets = dets[dets[:,4]>self.init_threshold]
            do_nms(dets,self.init_nms)
            dets= dets[dets[:,4]<1.1]
            dets= dets[dets[:,4]>0]

        # Create new trackers from unassigned detections
        if self.kalman_type == 'deepbeast':
            #here logic is a wee bit different, as the predictions are happening on the global level
            self.trackers = self.dbt_battery.create_trackers(dets)
        else:
            for det in dets:
                if self.kalman_type == 'sort':
                    trk = KalmanBoxSortTracker(det[:])
                elif self.kalman_type == 'torney':
                    trk = KalmanBoxTracker(det[:])
                else:
                    raise Exception(f'This should never happen, kalman type is {self.kalman_type}')
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        #Updated the dbt_battery trackers
        if self.kalman_type == 'deepbeast':
            self.dbt_battery.trackers = self.trackers

        # Provide an output list of trackers if needed/wanted
        for trk in (self.trackers):
            d = get_box(trk)
            # if ((trk.time_since_update < self.hold_without) and (trk.long_score>self.track_threshold)):
            #filtering out tracks can and should happen at later stage!
            ret.append(np.concatenate((d,[trk.id,trk.long_score,trk.score])).reshape(1,-1))

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

    def update(self,dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.

        #This function is broken down into two steps so it can be run separately if needed
        """
        self.update_0_predict()
        return self.update_1_update(dets)
