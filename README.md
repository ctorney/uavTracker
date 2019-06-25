# uavTracker
Animal tracking from overhead using YOLO


## Steps to create the tracker are 
### 1. Training
   Step 1 is to generate training samples. Code to generate samples is in the directory train.
  * To prepare samples we use the YOLOv3 pretrained weights on the images. This is done with  `prepTrain.py` code:
```
usage: prepTrain.py [-h] --config CONFIG --ddir DDIR

Detect objects in images using a pre-trained model, and prepare images for the
further processing. This program is used to pre-annotate images with a pre-
trained network (for instance yolo weights). It creates necessary output
directories and cuts your images to a given size and writes them to disk.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --ddir DDIR, -d DDIR  Root of your data directory

```
  
  * Once candidate detections are created the annotate.py can be used to filter out errors or objects that aren't of interest.
```
usage: annotate.py [-h] --config CONFIG --ddir DDIR [--resume]
                   [--from-scratch]

Annotate or correct annotations from non-domain specific model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --ddir DDIR, -d DDIR  Root of your data directory
  --resume, -r          Continue a session of annotation (it will access
                        output file and re-start when you left off)
  --from-scratch, -f    Annotate files from scratch without any existing
                        bounding boxes

```

   The final step is to train the model with the created training samples. This is done with train.py. We use freeze lower levels and use pretrained weights, then fine tune the whole network. In the config file you specify parameters for both _Phase 1_ and _Phase 2_ of the training with the defined files:

```
 usage: train.py [-h] --config CONFIG --ddir DDIR [--debug] [--test-run]
                --phase {phase_one,phase_two}

Fine tune a yolo model. You have to run phase_one, or phase_one followed by
phase_two. So basically you will run this program twice with different value
for --phase flag. Good luck!

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --ddir DDIR, -d DDIR  Root of your data directory
  --debug               Set this flag to see a bit more wordy output
  --test-run            Set this flag to see meaningless output quicker. For
                        instance in training it runs only for 1 epoch
  --phase {phase_one,phase_two}
                        phase_one: top layers, normal learning rate phase_two:
                        all layers, small learning rate.
```

Curious you might be of your results. The `postTrainTest.py` is used to prepare detections and original annotations in a form used by fantastic repo https://github.com/rafaelpadilla/Object-Detection-Metrics  which provides mAP and plots recall-precision curve.
```
usage: postTrainTest.py [-h] --config CONFIG --ddir DDIR

Prepare list of detection of original input files using final classifier.
Those files are saved in groundtruths and predictions directories which can be
interpreted by program https://github.com/rafaelpadilla/Object-Detection-
Metrics

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --ddir DDIR, -d DDIR  Root of your data directory
```

  
  

### Running tracker
To prepare transformations and a config file containing information parts of the clips which needs to be tracker run 
```
python3 transforms.py --d ../data/blackbucks -c ../blackbucks.yml
```
You can also prepare this file manually, an example is called `videos_template.yml`.

Then you can run tracker
```
python3 runTracker.py -d ../data/blackbucks -c ../blackbucks.yml
```

The working yml file is `blackbucks.yml` and `rockinghorse.yml`
You can now defined many different setups under config variable `training_setup`.

### Toy example:
The toy example is called rockinghorse, and the yml config file for it is in the root of this repository. In it we specify all the directories relative to where the yml file is. Then we call the annotation preparation or training as follows:

`python train.py -d ~/repos/uavTracker/data/rockinghorse/ -c rockinghorse.yml`

It is handy to store data (or symbolic link to it) in a folder `data` in the root directory, it is already added to `.gitignore`

### Tricks and treats:
* There is a script `utils/pickle2yaml.py` which converts your old pickles int new yaml.
