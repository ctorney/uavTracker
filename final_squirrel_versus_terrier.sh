set -e
rm -rf data/alfs*
cd utils
python toy_data_generator.py -c ../experiments/alfs_squirrel.yml
python toy_data_generator.py -c ../experiments/alfs_terrier.yml
cd ../data/alfs_squirrel
ln -s ../../weights .
cd ../alfs_terrier
ln -s ../../weights .

#terrier
cd ../../deepBeastLinker
python testTrackers.py -c ../experiments/alfs_terrier.yml
python testTrackers.py -c ../experiments/alfs_squirrel.yml
