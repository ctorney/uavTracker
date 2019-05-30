# uavTracker
Animal tracking from overhead using YOLO


## Steps to create the tracker are 
### 1. Training
   Step 1 is to generate training samples. Code to generate samples is in the directory train/prepare_samples.
  * To prepare samples we use the YOLOv3 pretrained weights on the images. This is done with the processImages.py code
  * Once candidate detections are created the manualOveride notebook can be used to filter out errors or objects that aren't of interest.
  * The final stage is to use the anchorboxes notebook to determine the bounding boxes to be used as a baseline for YOLO

   Step 2 is to train the model with the created training samples. This is done with train.py. We use freeze lower levels and use pretrained weights, then fine tune the whole network

### Toy example:
The toy example is called rockinghorse, and the yml config file for it is in the root of this repository. In it we specify all the directories relative to where the yml file is. Then we call the annotation preparation or training as follows:

`python prepTrain.py [data_directory] [configfile_name.yml]`

`python train.py [data_directory] [configfile_name.yml]`

for instance:

`python train.py ~/repos/uavTracker/data/rockinghorse/ rockinghorse.yml`

It is handy to store data (or symbolic link to it) in a folder `data` in the root directory, it is already added to `.gitignore`

### Tricks and treats:
* There is a script `utils/pickle2yaml.py` which converts your old pickles int new yaml.
