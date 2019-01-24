"""
This little helpful fellow will convert your nasty pickles to our nice, text-readable yaml"""
import pickle
import yaml
import sys


def main(argv):
    if(len(sys.argv) != 3):
        print('Usage python pickle2yaml.py monster.pickle beauty.yml')
        sys.exit(1)

    #Load data
    print('Opening pickle file' + argv[1])
    with open(argv[1], 'rb') as monster:
        content = pickle.load(monster)

    print('Writing to yaml file' + argv[2])
    with open(argv[2], 'w') as beauty:
        yaml.dump(content, beauty)

    print('Finito amigo!')

if __name__ == '__main__':
    main(sys.argv)
