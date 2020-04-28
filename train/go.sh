set -e
python train.py --config ../alfs.yml --ddir ../data/alfs --phase='phase_one'
python train.py --config ../alfs.yml --ddir ../data/alfs --phase='phase_two'

