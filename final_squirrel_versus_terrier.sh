set -e
rm -rf data/alfs*
#rm -f weights/alfs_*
cd utils
python toy_data_generator.py -c ../experiments/alfs_squirrel.yml
python toy_data_generator.py -c ../experiments/alfs_terrier.yml
cd ../data/alfs_squirrel
ln -s ../../weights .
cd ../alfs_terrier
ln -s ../../weights .

#terrier
cd ../../train
python train.py -c ../experiments/alfs_terrier.yml
python postTrainTest.py -c ../experiments/alfs_terrier.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_terrier.yml
python testTrackers.py -c ../experiments/alfs_terrier.yml

#squirrel
cd ../train
python train.py -c ../experiments/alfs_squirrel.yml
python postTrainTest.py -c ../experiments/alfs_squirrel.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_squirrel.yml
python testTrackers.py -c ../experiments/alfs_squirrel.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_terrier.yml -s
python runTracker.py -c ../experiments/alfs_terrier.yml -s
python transforms.py -c ../experiments/alfs_squirrel.yml -s
python runTracker.py -c ../experiments/alfs_squirrel.yml -s
