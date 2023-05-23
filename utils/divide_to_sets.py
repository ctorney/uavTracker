import os, glob, shutil
from numpy.random import default_rng

all_folder = '../data/gnu/subsets/train_old/'
imgs_to_pull = 2000
output_folder = '../data/gnu/subsets/test_old/'
allfiles =glob.glob(f'{all_folder}/*.JPG')

rng = default_rng()
for iii in rng.choice(len(allfiles),imgs_to_pull, replace=False):
    shutil.move(allfiles[iii], os.path.join(output_folder,os.path.basename(allfiles[iii])))
