# uavTracker
Animal tracking from overhead using YOLO

Try the following example
```
python train.py --config ../experiments/easy_fish.yml --test-run
```

### Training/testing sets
You can specify any number of training and testing sets that will be used as described in the experiment config file. Just take a look at the provided example.
There are two ways of engagin with datasets:
a) You already have an annotations file with **unique** filenames for every image in different subset directories. We are scanning directories of subsets as prescribed and pulling the correct annotations from that file, creating a temporary annotations file that contains specificed subset for either training or testing.

b) You will annotate the data using our system, in which case, the filenames can be repeating across different subsets because we will include the subset directory as their names. I still think the names of images should be unique, but I won't be adding a checksum for every image to make sure it is the same :D




# Old instructions to be removed or updated

To check if it works run the following script which downloads some pretrained model and data and makes a test-run at re-training and tracking. It is interactive so at some point expect to annotate files - if you don't have X-server then you might need to remove this step and manually create a checked annotations file as per `rockinghorse.yml` config
```
export WHICH_PYTHON=python3 && ./test.sh
```

## Steps to create the tracker are
### 0. pre-training
   This pre-training is usually necessary as a way to clear/check or create from scratch annotations for your target animal. We leverage often decent performance of a generic object detector to generate imperfect annotated data (prepTrain) and then allow corrections (annotate.py). It is also known as transfer learning because we are not training (train.py) from scratch but take an existing model.
   Step 0 is to generate training samples. Code to generate samples is in the directory train.
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

Annotate or correct annotations from non-domain specific model.To remove annotation double left click. To add one, Middle Click and move. \'c\' accepts changes and goes to the next image, \'q\' ends the session and saves files done so far (resume option is used to continue this work).

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

### 1. Training
   This is where everything gets exciting once you have annotated images.
   The key step is to train the model with the created training samples. This is done with train.py. We use freeze lower levels and use pretrained weights, then fine tune the whole network. In the config file you specify parameters for both _Phase 1_ and _Phase 2_ of the training with the defined files:

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
