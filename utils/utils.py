import hashlib, math, yaml, glob, cv2, sys, os
import numpy as np
from scipy.special import expit

"""
A little function to provide all of the parameters passed as arguments
"""
def init_config(
        args={'config': ['../experiments/toys.yml'],
              'debug': True,
              'visual': True,
              'tracker': False,
              'annotated': False,
              'test_run': False,
              'output': ['tmp.txt'],
              'resume': False,
              'from_scratch': False,
              }):
    #convert this stupid Namespace object to a normal cuddly dictionary
    if type(args)!=dict:
        args = vars(args)

    print('Opening file' + args['config'][0])
    with open(args['config'][0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    #supplement config with arguments provided in the commandline, for ease of use
    config['args_debug'] = args['debug'] if ('debug' in args.keys()) else False
    config['args_annotated'] = args['annotated'][0] if ('annotated' in args.keys() and type(args['annotated'])==list) else False
    config['args_visual'] = args['visual'] if ('visual' in args.keys()) else False
    config['args_resume'] = args['resume'] if ('resume' in args.keys()) else False
    config['args_from_scratch'] = args['from_scratch'] if ('from_scratch' in args.keys()) else False
    config['args_tracker'] = args['tracker'] if ('tracker' in args.keys()) else False
    config['args_step'] = args['step'] if ('step' in args.keys()) else False
    config['args_output'] = args['output'][0] if ('output' in args.keys() and type(args['output'])==list) else False
    config['args_test_run'] = args['test_run'] if ('test_run' in args.keys()) else False

    if config['args_debug']:
        print(config)

    return config


"""
Run through the existing annotations and create a list of files without any annotations.
checked(manual) and pre(auto)

Returns a dictionary where keys are directory paths and values are lists of filenames from those directories that need to be annotated
"""
def filter_out_annotations(ss_imgs_all,some_exists,annotations_file):
    all_annot_imgs = []
    to_annot_dict = dict()

    if some_exists:
        print(f"Loading annotated images annotations from {annotations_file}")
        with open(annotations_file, 'r') as fp:
            all_annot_imgs = all_annot_imgs + yaml.safe_load(fp)

        annot_filenames = []
        for annotation_data in all_annot_imgs:
            annot_filenames.append(annotation_data['filename'])

        for ssdir, ssilist in ss_imgs_all.items():
            to_annot_dict[ssdir]=[]
            for ssi in ssilist:
                if not ssi in annot_filenames:
                    to_annot_dict[ssdir].append(ssi)
    else:
        to_annot_dict = ss_imgs_all

    return to_annot_dict


"""
Reads all subsets (defined as directories with file lists or regex) provided in a config providing a list of paths to the files.
Returns a list of strings
"""
def read_subsets(list_of_subsets,config):
    ss_imgs_all = dict()
    subsets = config['subsets']

    for tset in list_of_subsets:
        tdir = os.path.join(config['project_directory'],subsets[tset]['directory'])
        #this is risky....
        #If the directory for the subset doesn't exist, use the unprocessed/raw directory
        if not os.path.exists(tdir):
            print('Using raw directory. This should only really happen in prepTrain.py function, otherwise it means that there are still images unadapted to yolo and rather unannotated')
            tdir = os.path.join(os.path.join(config['project_directory'],config['raw_imgs_dir']), subsets[tset]['directory'])

        tnimg = subsets[tset]['number_of_images']
        if subsets[tset]['filelist']:
            with open(tdir + subsets[tset]['filelist'], 'r') as fl:
                ss_imgs = fl.read().splitlines()
            for fname in ss_imgs:
                if not os.path.exists(tdir + fname):
                    raise Exception(f'ERROR: In {tset} file {fname} is missing but according to the filelist it should be ther')
        elif subsets[tset]['regex']:
            ss_imgs_full = glob.glob(tdir + subsets[tset]['regex'])
            ss_imgs = [os.path.basename(x) for x in ss_imgs_full]
        else:
            raise Exception(f'ER R R R O RRRR: the subset {tset} has no filelist or regex defined. The simples regex would be \'*\' to include all files in the directory')

        if len(ss_imgs) != tnimg:
            lss_imgs = len(ss_imgs)
            raise Exception(f'Error: for the subset {tset} number of images defined {tnimg} is not matching the provided filename or regex {lss_imgs}. Maybe you have partially run prepTrain.py and only some of the files from unprocessed directory got moved into the main project directory?')
        ss_imgs_all[tdir] = ss_imgs

    return ss_imgs_all

"""
Read a subset of the training/testing set for the given model to train from the config. Following that prepare a temporary annotations file that contains those specified image files with their annotations only.

Returns:
Filename of temporary annotations file
"""
def read_tsets(config, model_name, c_date, list_of_subsets):
    subsets = config['subsets']

    all_annotations = os.path.join(config['project_directory'], config['annotations_dir'], config['checked_annotations_fname'])
    md5check(config['checked_annotations_md5'], all_annotations)

    print(f"Loading all images annotations from {all_annotations}")
    with open(all_annotations, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    ss_imgs_all = read_subsets(list_of_subsets, config)

    #Re-write existing annotations but with a subset path in a filename into a new file that will be used for this training
    annotations_subset = []
    for annotation_data in all_imgs:
        fname = annotation_data['filename']
        #if fname is in our set of filenames. But what if filenames are repeating?
        addit = False
        for ssdir, ssi in ss_imgs_all.items():
            if fname in ssi:
                annotation_data['filename'] = os.path.join(ssdir, fname)
                addit = True
                break
        if addit:
            annotations_subset += [annotation_data]


    current_annotations = os.path.join(config['project_directory'], config['annotations_dir'], 'annotations_' + model_name + '_' + c_date + '.yml')
    with open(current_annotations, 'w') as handle:
        yaml.dump(annotations_subset, handle)

    return current_annotations

"""
Provide a checksum for an edited file
"""
def getmd5(my_file):
    if os.path.exists(my_file):
        read_file_hex = hashlib.md5(open(my_file,'rb').read()).hexdigest()
        print(f" (new) MD5 check on your file {my_file}: is {read_file_hex}, consider amending it in the config")
        return read_file_hex

"""
Return True if file exists, False if it isn't. Prints helpful information and throws and exception if the md5 checksum do not check out.
"""
def md5check(md5sum,my_file):
    if os.path.exists(my_file):
        if md5sum != "":
            read_file_hex = hashlib.md5(open(my_file,'rb').read()).hexdigest()
            if read_file_hex != md5sum:
                raise Exception(f"ERROR: md5 checksum of the provided {my_file} file doesn't match!")
            else:
                print(f"MD5 check on your file {my_file}: correct! :D ")
        else:
            print(":: Kind notice :: No md5 sum provided. Consider adding it once you have some trained weights / annotated files whatever it is that you are doing to avoid great confusion in the future")
        return True
    else:
        return False

"""
Cut the image to a multiply of 32
"""
def makeYoloCompatible(image):
    im_height, im_width = image.shape[:2]
    new_width = 32 * math.floor(im_width / 32)
    new_height = 32 * math.floor(im_height / 32)
    im_yolo = image[0:new_height,0:new_width,:]
    return im_yolo

def pleaseCheckMyDirectories(config, data_dir):
    print('Checking if all output directories exist alright')

    groundtruths_dir = os.path.join(data_dir, config['groundtruths_dir'])
    predictions_dir = os.path.join(data_dir, config['predictions_dir'])
    annotations_dir = os.path.join(data_dir, config['annotations_dir'])
    bbox_images_dir = os.path.join(data_dir, config['bbox_images_dir'])
    tracks_dir = os.path.join(data_dir, config['tracks_dir'])

    dirs = [
        groundtruths_dir, predictions_dir, annotations_dir, bbox_images_dir, tracks_dir
    ]

    for dir in dirs:
        if not os.path.isdir(dir):
            print('Creating ' + dir)
            os.makedirs(dir, exist_ok=True)

def save_pascal_gt_detect_file(fname_gt, boxes_gt, args_visual, frame, img_data):
    with open(fname_gt, 'w') as file_gt:  #left top righ bottom
        for b in boxes_gt:
            obj = {}
            if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                continue
            obj['name'] = 'aoi'
            obj['xmin'] = int(b[0])
            obj['ymin'] = int(b[1])
            obj['xmax'] = int(b[2])
            obj['ymax'] = int(b[3])
            img_data['object'] += [obj]
            file_gt.write(obj['name'] + " ")
            file_gt.write(str(obj['xmin']) + " ")
            file_gt.write(str(obj['ymin']) + " ")
            file_gt.write(str(obj['xmax']) + " ")
            file_gt.write(str(obj['ymax']))
            file_gt.write('\n')

            if args_visual:
                cv2.rectangle(
                    frame, (int(obj['xmin']) - 2, int(obj['ymin']) - 2),
                    (int(obj['xmax']) + 2, int(obj['ymax']) + 2), (200, 0, 0), 1)
    return frame

def from_annot_save_pascal_pred_detect_file(pred_imgs,i,fname_pred,args,frame,max_l, min_l, im_size_w, im_size_h):
    boxes_pred = []
    for obj in pred_imgs[i]['object']:
        boxes_pred.append(
            [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])


    frame = save_pascal_pred_detect_file(boxes_pred,i,fname_pred,args,frame,max_l, min_l, im_size_w, im_size_h)
    return frame

def save_pascal_pred_detect_file(boxes_predict,i,fname_pred,args_visual,frame,max_l, min_l, im_size_w, im_size_h):
    with open(fname_pred, 'w') as file_pred:  #left top righ bottom
        for b in boxes_predict:
            xmin = int(b[0])
            xmax = int(b[2])
            ymin = int(b[1])
            ymax = int(b[3])
            confidence = float(b[4])
            objpred = {}

            objpred['name'] = 'aoi'

            if xmin < 0: continue
            if ymin < 0: continue
            if xmax > im_size_w: continue
            if ymax > im_size_h: continue
            if (xmax - xmin) < min_l: continue
            if (xmax - xmin) > max_l: continue
            if (ymax - ymin) < min_l: continue
            if (ymax - ymin) > max_l: continue

            objpred['xmin'] = xmin
            objpred['ymin'] = ymin
            objpred['xmax'] = xmax
            objpred['ymax'] = ymax
            objpred['confidence'] = confidence
            file_pred.write(objpred['name'] + " ")
            file_pred.write(str(objpred['confidence']) + " ")
            file_pred.write(str(objpred['xmin']) + " ")
            file_pred.write(str(objpred['ymin']) + " ")
            file_pred.write(str(objpred['xmax']) + " ")
            file_pred.write(str(objpred['ymax']))
            file_pred.write('\n')

            if args_visual:
                cv2.rectangle(
                    frame, (int(objpred['xmin']) - 2, int(objpred['ymin']) - 2),
                    (int(objpred['xmax']) + 2, int(objpred['ymax']) + 2), (0, 0, 198), 1)
                str_conf = "{:.1f}".format(objpred['confidence'])
                cv2.putText(frame, str_conf,  (int(objpred['xmax']),int(objpred['ymax'])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1);
    return frame
