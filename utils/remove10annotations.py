import os, sys, argparse, yaml
from numpy.random import default_rng
from numpy import arange

def main(args):

    rng = default_rng()
    config_file = args.config[0]
    with open(config_file, 'r') as configfile:
        config = yaml.safe_load(configfile)
    ddir = config['project_directory']

    checked_annotations = os.path.join(config['project_directory'],config['annotations_dir'],config['checked_annotations_fname'])

    with open(checked_annotations, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    #the most horrific way
    new_imgs = [all_imgs[i] for i in list(set(arange(len(all_imgs))) - set(rng.choice(len(all_imgs), 10, replace=False)))]

    with open(checked_annotations, 'w') as handle:
        yaml.dump(new_imgs, handle)

    print('You should have 10 images to annotated now')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        'Remove 10 annotations to do some work by yourself',
        epilog=
        'Any issues and clarifications: github')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')

    args = parser.parse_args()

    main(args)
