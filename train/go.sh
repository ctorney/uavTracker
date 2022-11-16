set -e
python train.py --config ../alfs.yml --ddir ../data/alfs_shapeshifters --phase='phase_one'
python train.py --config ../alfs.yml --ddir ../data/alfs_shapeshifters --phase='phase_two'

