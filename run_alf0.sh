set -e
rm -rf data/final_alfsanity
rm -f weights/alfsanity_terrier*
cd utils
python toy_data_generator.py -c ../experiments/alfsanity.yml -g ../experiments/alfsanity_datasets.yml
cd ../data/final_alfsanity
ln -s ../../weights .
cd ../../train
python train.py -c ../experiments/alfsanity.yml --test-run
python postTrainTest.py -c ../experiments/alfsanity.yml
cd ../tracking
python transforms.py -c ../experiments/alfsanity.yml -s
python runTracker.py -c ../experiments/alfsanity.yml -s
cd ../deepBeastLinker
python trainTracker.py -c ../experiments/alfsanity.yml

