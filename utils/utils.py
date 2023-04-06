import cv2
import numpy as np
import sys
import os
from scipy.special import expit
import hashlib, math, yaml, glob



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
