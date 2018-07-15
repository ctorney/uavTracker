# deepTracker
Animal tracking from overhead using YOLO


## Steps to create the tracker are 
### 1. Training
   Step 1 is to generate training samples. Code to generate samples is in the directory train/prepare_samples.
  * To prepare samples we use the YOLOv3 pretrained weights on the images. This is done with the processImages.py code
  * Once candidate detections are created the manualOveride notebook can be used to filter out errors or objects that aren't of interest.
  * The final stage is to use the anchorboxes notebook to determine the bounding boxes to be used as a baseline for YOLO

   Step 2 is to train the model with the created training samples. This is done with train.py. We use freeze lower levels and use pretrained weights, then fine tune the whole network
