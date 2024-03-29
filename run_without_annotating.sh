set -e
rm -rf data/toys
rm -f weights/toys_model_1_phase_*
cd utils
python toy_data_generator.py -c ../experiments/toys.yml -g ../experiments/toys_generator.yml
cd ../data/toys
ln -s ../../weights .
cd ../../train
python train.py -c ../experiments/toys.yml 
python postTrainTest.py -c ../experiments/toys.yml
