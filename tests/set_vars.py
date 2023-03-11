import yaml

data_dir = '../data/alfs_shapeshifters/'
data_dir = '../data/alfs_gt/'
args_visual = False
args_annotated = False



### new
conf_n = '../experiments/easy_fish.yml'
conf_n = '../experiments/alfs_shapeshift.yml'
conf_n = '../experiments/alfs_shapeshift_tracking.yml'
with open(conf_n, 'r') as configfile:
    config = yaml.safe_load(configfile)
DEBUG = False
TEST_RUN=True

model_name = 'model_1'
data_dir = '../data/easy_fish/'
args_visual = True

list_of_subsets = c_model['training_sets']
