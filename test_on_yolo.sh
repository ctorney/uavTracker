#!/usr/bin/env bash
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::                                                                ::"
echo "Testing rocking horse on yolo weghts. Make sure to copy yolo-v3-coco.h5 to the directory for weights in rockinghorse!"
echo "::                                                                ::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
set -ex
if [ ! -f rockinghorse.zip ]; then
	wget https://www.dropbox.com/s/fredcwheohguypa/rockinghorse.zip
	unzip rockinghorse.zip
fi
cd train
python prepTrain.py --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse
python annotate.py --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse
#--test-run flag runs training for a test run
python train.py --test-run --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse --phase='phase_one'
python train.py --test-run --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse --phase='phase_two'
python postTrainTest.py --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse
cd ../tracking
python transforms.py --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse
python runTracker.py --config ../rockinghorse_fromyolo.yml --ddir ../rockinghorse
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
echo "It all seem to be working... Testing script completed. Ufff!"
echo "::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::>- O O -<::"
