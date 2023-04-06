#!/bin/bash
set -e
rm -r data/toys
cd utils
python toy_data_generator.py -c ../experiments/toys.yml --size 500
cd ../data/toys
ln -s ../../weights .
cd ../../utils
python remove10annotations.py -c ../experiments/toys.yml
cd ..
