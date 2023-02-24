# uavTracker
Animal tracking from overhead using YOLO

Try the following example
```
python prepTrain.py -c ../experiments/easy_fish.yml
python annotate.py -c ../experiments/easy_fish.yml
python train.py --config ../experiments/easy_fish.yml
python postTrainTest.py --config ../experiments/easy_fish.yml
```
## Steps to create the tracker are
### Training/testing sets
You can specify any number of training and testing sets that will be used as described in the experiment config file. Just take a look at the provided example.
There are two ways of engaging with datasets:
a) You already have an annotations file with **unique** filenames for every image in different subset directories. We are scanning directories of subsets as prescribed and pulling the correct annotations from that file, creating a temporary annotations file that contains specificed subset for either training or testing.

b) You will annotate the data using our system, in which case, the filenames can be repeating across different subsets because we will include the subset directory as their names. I still think the names of images should be unique, but I won't be adding a checksum for every image to make sure it is the same :D
### Prepare directory

The directory needs to match the provided config file. It is best to learn from looking at the example `experiments/easy_fish.yml`. The essential thing is to provide images for training.
If your images are unprocessed, i.e. have some unknown size, you need to create a directory defined in a config as `raw_imgs_dir` that will be a root for datasets defined in a config. Once the `prepTrain.py` script runs, it will copy the images with sizes adjusted to YOLO into the main directory of the project. For instance a subset `directory` key defines `directory: subsets/setA`, and `raw_imgs_dir: rawimg`, in this case put your images of Set A into `rawimg/subsets/setA`

If your images are already right sizes, you can put them in the root directory of the project straight away.

Note:
*prepTrain doesn't copy files that already have annotations from "raw" directory to the main one.*

It makes sense to create a symbolic link to your directory with weights in the data folder of your project

Here is an example of a directory tree for a config file `/experiments/easy_fish.yml`:
```
data/easy_fish
├── raw_imgs
│   └── subsets
│       ├── test_A
│       │   ├── TR_f900f-0.png
│       │   ├── TR_f9100f-0.png
│       │   └── TR_f9300f-0.png
│       ├── test_B
│       │   ├── test_b_filelist.txt
│       │   ├── TR_vlcsnap-2019-03-10-22h57m51s173-0.png
│       │   └── TR_vlcsnap-2019-03-10-22h57m52s331-0.png
│       ├── training_A
│       │   ├── TR_f10400f-0.png
│       │   ├── TR_f1100f-0.png
│       │   └── TR_f1400f-0.png
│       └── training_B
│           ├── TR_f1800f-0.png
│           ├── TR_f21000f-0.png
│           ├── TR_f24900f-0.png
│           ├── TR_f27300f-0.png
│           ├── TR_f30200f-0.png
│           ├── TR_f31000f-0.png
│           ├── TR_f33800f-0.png
│           ├── TR_f37300f-0.png
│           └── TR_f5300f-0.png
└── weights
    ├── best_smolts_standard_phase_two_2022Dec05.h5
    └── yolo-v3-coco.h5
```
and after running of the detection training and testing
```
data/easy_fish
├── annotations
│   ├── annotations_model_1_2023Feb01.yml
│   ├── annotations_model_2_2023Feb01.yml
│   ├── annotations_testing_2023Feb01.yml
│   ├── autogenerated_annotations.yml
│   └── manual_annotatations.yml
├── extracted_tracks
├── raw_imgs
│   └── subsets
│       ├── test_A
│       │   ├── TR_f900f-0.png
│       │   ├── TR_f9100f-0.png
│       │   └── TR_f9300f-0.png
│       ├── test_B
│       │   ├── test_b_filelist.txt
│       │   ├── TR_vlcsnap-2019-03-10-22h57m51s173-0.png
│       │   └── TR_vlcsnap-2019-03-10-22h57m52s331-0.png
│       ├── training_A
│       │   ├── TR_f10400f-0.png
│       │   ├── TR_f1100f-0.png
│       │   └── TR_f1400f-0.png
│       └── training_B
│           ├── TR_f1800f-0.png
│           ├── TR_f21000f-0.png
│           ├── TR_f24900f-0.png
│           ├── TR_f27300f-0.png
│           ├── TR_f30200f-0.png
│           ├── TR_f31000f-0.png
│           ├── TR_f33800f-0.png
│           ├── TR_f37300f-0.png
│           └── TR_f5300f-0.png
├── results
│   ├── groundtruths
│   │   ├── data-easy_fish-subsets-test_A-TR_f900f-0.png.txt
│   │   ├── data-easy_fish-subsets-test_A-TR_f9100f-0.png.txt
│   │   ├── data-easy_fish-subsets-test_A-TR_f9300f-0.png.txt
│   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m51s173-0.png.txt
│   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m52s331-0.png.txt
│   │   ├── data-easy_fish-subsets-training_A-TR_f10400f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_A-TR_f1100f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_A-TR_f1400f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f1800f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f21000f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f24900f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f27300f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f30200f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f31000f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f33800f-0.png.txt
│   │   ├── data-easy_fish-subsets-training_B-TR_f37300f-0.png.txt
│   │   └── data-easy_fish-subsets-training_B-TR_f5300f-0.png.txt
│   ├── predictions
│   │   ├── model_1_phase_one_2023Feb01
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f900f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f9100f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f9300f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m51s173-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m52s331-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f10400f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f1100f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f1400f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f1800f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f21000f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f24900f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f27300f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f30200f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f31000f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f33800f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f37300f-0.png.txt
│   │   │   └── data-easy_fish-subsets-training_B-TR_f5300f-0.png.txt
│   │   ├── model_1_phase_two_2023Feb01
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f900f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f9100f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_A-TR_f9300f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m51s173-0.png.txt
│   │   │   ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m52s331-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f10400f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f1100f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_A-TR_f1400f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f1800f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f21000f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f24900f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f27300f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f30200f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f31000f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f33800f-0.png.txt
│   │   │   ├── data-easy_fish-subsets-training_B-TR_f37300f-0.png.txt
│   │   │   └── data-easy_fish-subsets-training_B-TR_f5300f-0.png.txt
│   │   └── model_2_phase_one_2023Feb01
│   │       ├── data-easy_fish-subsets-test_A-TR_f900f-0.png.txt
│   │       ├── data-easy_fish-subsets-test_A-TR_f9100f-0.png.txt
│   │       ├── data-easy_fish-subsets-test_A-TR_f9300f-0.png.txt
│   │       ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m51s173-0.png.txt
│   │       ├── data-easy_fish-subsets-test_B-TR_vlcsnap-2019-03-10-22h57m52s331-0.png.txt
│   │       ├── data-easy_fish-subsets-training_A-TR_f10400f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_A-TR_f1100f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_A-TR_f1400f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f1800f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f21000f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f24900f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f27300f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f30200f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f31000f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f33800f-0.png.txt
│   │       ├── data-easy_fish-subsets-training_B-TR_f37300f-0.png.txt
│   │       └── data-easy_fish-subsets-training_B-TR_f5300f-0.png.txt
│   └── results_file.yml
├── subsets
│   ├── test_A
│   │   ├── TR_f900f-0.png
│   │   ├── TR_f9100f-0.png
│   │   └── TR_f9300f-0.png
│   ├── test_B
│   │   ├── test_b_filelist.txt
│   │   ├── TR_vlcsnap-2019-03-10-22h57m51s173-0.png
│   │   └── TR_vlcsnap-2019-03-10-22h57m52s331-0.png
│   ├── training_A
│   │   ├── TR_f10400f-0.png
│   │   ├── TR_f1100f-0.png
│   │   └── TR_f1400f-0.png
│   └── training_B
│       ├── TR_f1800f-0.png
│       ├── TR_f21000f-0.png
│       ├── TR_f24900f-0.png
│       ├── TR_f27300f-0.png
│       ├── TR_f30200f-0.png
│       ├── TR_f31000f-0.png
│       ├── TR_f33800f-0.png
│       ├── TR_f37300f-0.png
│       └── TR_f5300f-0.png
└── weights
    ├── best_smolts_standard_phase_two_2022Dec05.h5
    └── yolo-v3-coco.h5
```
### Preparing for training
   This pre-training is usually necessary as a way to clear/check or create from scratch annotations for your target animal. We leverage often decent performance of a generic object detector to generate imperfect annotated data (prepTrain) and then allow corrections (annotate.py). It is also known as transfer learning because we are not training (train.py) from scratch but take an existing model.
   Step 0 is to generate training samples. Code to generate samples is in the directory train.
  * To prepare samples we use the YOLOv3 pretrained weights on the images. This is done with  `prepTrain.py` code:
```
usage: prepTrain.py [-h] --config CONFIG

Detect objects in images using a pre-trained model, and prepare images for the
further processing. This program is used to pre-annotate images with a pre-
trained network (for instance yolo weights). It creates necessary output
directories and cuts your images to a given size and writes them to disk.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
```

  * Once candidate detections are created the annotate.py can be used to filter out errors or objects that aren't of interest.

```
usage: annotate.py [-h] --config CONFIG
                   [--from-scratch]

Annotate or correct annotations from non-domain specific model.To remove annotation double left click. To add one, Middle Click and move. \'c\' accepts changes and goes to the next image, \'q\' ends the session and saves files done so far (resume option is used to continue this work).

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --resume, -r          Continue a session of annotation (it will access
                        output file and re-start when you left off)
  --from-scratch, -f    Annotate files from scratch without any existing
                        bounding boxes

```

### Training
   This is where everything gets exciting once you have annotated images.
   The key step is to train the model with the created training samples. This is done with train.py. We use freeze lower levels and use pretrained weights, then fine tune the whole network. In the config file you specify parameters for both _Phase 1_ and _Phase 2_ of the training with the defined files:

```
 usage: train.py [-h] --config CONFIG [--debug] [--test-run]

Fine tune a yolo model. You have to run phase_one, or phase_one followed by
phase_two. So basically you will run this program twice with different value
for --phase flag. Good luck!

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
  --debug               Set this flag to see a bit more wordy output
  --test-run            Set this flag to see meaningless output quicker. For
                        instance in training it runs only for 1 epoch
  --phase {phase_one,phase_two}
                        phase_one: top layers, normal learning rate phase_two:
                        all layers, small learning rate.
```

Curious you might be of your results. The `postTrainTest.py` is used to prepare detections and original annotations in a form used by fantastic repo https://github.com/rafaelpadilla/Object-Detection-Metrics  which provides mAP and plots recall-precision curve.
```
usage: postTrainTest.py [-h] --config CONFIG

Prepare list of detection of original input files using final classifier.
Those files are saved in groundtruths and predictions directories which can be
interpreted by program https://github.com/rafaelpadilla/Object-Detection-
Metrics

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Your yml config file
```


### Running tracker

Put your videos in a directory specified by regex. Preparing camera transformations also prepares lists of files to tracking. Refer to `experiments/alfs23_tracking.yml` to see example

`python transform.py -c ../experiments/alfs23_tracking.yml`
`python runTracker.py -c ../experiments/alfs23_tracking.yml --visual`

### Correcting tracks
Run the following program that displays (on top) current output of tracking as produced in the previous section and (below) corrected tracks. You can scroll through a video (with a limited buffer) with letters `d` (forward) and `a` (back). Press `l` to reload the `transitions` and `switches` files and `q` to close. Transitions is a file generated in a previous step when in each line you provide a list of different IDs that one track assumes. For instance
```
1,4,8
```
means that track number 1, changes to 4 and 8 but is essentially the same track and we want it to be ID as 1.
In the next step provide the switches, which means that two tracks are completely swapping from a given frame onwards (one switch per line):
```
10,1,12
```
means that from frame 10 onwards track 1 and 12 need to be switched around.

The transitions file is applied first so "switches" are applied after that corrections

After a full run the program produces a `_corrected` file with positions in the same format as the original output with your corrections incorporated

`python correctTracks.py -c ../experiments/alfs23_tracking.yml`
