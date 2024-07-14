set -e
rm -rf data/final_alfs_squirrel
rm -f weights/alfs_squirrel*
cd utils
python toy_data_generator.py -c ../experiments/alfs_squirrel.yml -g ../experiments/alfs_datasets.yml
cd ../data/final_alfs_squirrel
ln -s ../../weights .
cd ../../train
python train.py -c ../experiments/alfs_squirrel.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_squirrel.yml
python testTrackers.py -c ../experiments/alfs_squirrel.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_squirrel.yml -s
python runTracker.py -c ../experiments/alfs_squirrel.yml -s
cd ../train
python postTrainTest.py -c ../experiments/alfs_squirrel.yml
