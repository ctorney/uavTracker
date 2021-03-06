#!/usr/bin/env bash
set -ex
echo "Using python:"
$WHICH_PYTHON -V
echo "Downloading all libraries, please wait :) "
$WHICH_PYTHON -m pip install -r requirements.txt
if [ ! -f rockinghorse.zip ]; then
	wget https://www.dropbox.com/s/fredcwheohguypa/rockinghorse.zip
	unzip rockinghorse.zip
fi
cd train
$WHICH_PYTHON prepTrain.py --config ../rockinghorse.yml --ddir ../rockinghorse
$WHICH_PYTHON annotate.py --config ../rockinghorse.yml --ddir ../rockinghorse
#--test-run flag runs training for a test run
$WHICH_PYTHON train.py --test-run --config ../rockinghorse.yml --ddir ../rockinghorse --phase='phase_one'
$WHICH_PYTHON train.py --test-run --config ../rockinghorse.yml --ddir ../rockinghorse --phase='phase_two'
$WHICH_PYTHON postTrainTest.py --config ../rockinghorse.yml --ddir ../rockinghorse
cd ../tracking
$WHICH_PYTHON transforms.py --config ../rockinghorse.yml --ddir ../rockinghorse
$WHICH_PYTHON runTracker.py --config ../rockinghorse.yml --ddir ../rockinghorse
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
echo "It all seem to be working... Testing script completed. Ufff!"
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
