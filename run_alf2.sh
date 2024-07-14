set -e
rm -rf data/final_alfs
rm -f weights/alfs_terrier*
cd utils
python toy_data_generator.py -c ../experiments/alfs.yml -g ../experiments/alfs_datasets.yml
cd ../data/final_alfs
ln -s ../../weights .
cd ../../train
python train.py -c ../experiments/alfs.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs.yml
python testTrackers.py -c ../experiments/alfs.yml
cd ../tracking
python transforms.py -c ../experiments/alfs.yml -s
python runTracker.py -c ../experiments/alfs.yml -s
cd ../train
python postTrainTest.py -c ../experiments/alfs.yml
