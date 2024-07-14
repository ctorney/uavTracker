set -e
rm -rf data/alfs*
rm -f weights/alfs_*
cd utils
python toy_data_generator.py -c ../experiments/alfs_squirrel.yml
python toy_data_generator.py -c ../experiments/alfs_squirrel_identical.yml
python toy_data_generator.py -c ../experiments/alfs_terrier.yml
python toy_data_generator.py -c ../experiments/alfs_terrier_identical.yml
cd ../data/alfs_squirrel
ln -s ../../weights .
cd ../data/alfs_terrier
ln -s ../../weights .
cd ../data/alfs_squirrel_identical
ln -s ../../weights .
cd ../data/alfs_terrier_identical
ln -s ../../weights .
#squirrel
cd ../../train
python train.py -c ../experiments/alfs_squirrel.yml
cd ../train
python postTrainTest.py -c ../experiments/alfs_squirrel.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_squirrel.yml
python testTrackers.py -c ../experiments/alfs_squirrel.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_squirrel.yml -s
python runTracker.py -c ../experiments/alfs_squirrel.yml -s

#terrier
cd ../../train
python train.py -c ../experiments/alfs_terrier.yml
cd ../train
python postTrainTest.py -c ../experiments/alfs_terrier.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_terrier.yml
python testTrackers.py -c ../experiments/alfs_terrier.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_terrier.yml -s
python runTracker.py -c ../experiments/alfs_terrier.yml -s


#squirrel
cd ../../train
python train.py -c ../experiments/alfs_squirrel_identical.yml
cd ../train
python postTrainTest.py -c ../experiments/alfs_squirrel_identical.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_squirrel_identical.yml
python testTrackers.py -c ../experiments/alfs_squirrel_identical.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_squirrel_identical.yml -s
python runTracker.py -c ../experiments/alfs_squirrel_identical.yml -s

#terrier
cd ../../train
python train.py -c ../experiments/alfs_terrier_identical.yml
cd ../train
python postTrainTest.py -c ../experiments/alfs_terrier_identical.yml
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfs_terrier_identical.yml
python testTrackers.py -c ../experiments/alfs_terrier_identical.yml
cd ../tracking
python transforms.py -c ../experiments/alfs_terrier_identical.yml -s
python runTracker.py -c ../experiments/alfs_terrier_identical.yml -s
