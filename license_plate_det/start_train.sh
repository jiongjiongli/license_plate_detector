cd /project/train/src_repo/ultralytics
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/ultralytics

echo 'Reset env...'
mkdir -p /project/train/models

rm -rf /project/train/tensorboard/*
mkdir -p /project/train/tensorboard

rm -rf /project/train/log/*
mkdir -p /project/train/log

rm -rf /project/train/result-graphs/*
mkdir -p /project/train/result-graphs

# python misc/data_analyzer.py

echo 'Start data_config...'
python license_plate_det/data_config.py

# echo 'Start yolo_tune...'
# python license_plate_det/yolo_tune.py

echo 'Start yolo_train...'
python license_plate_det/yolo_train.py

# echo 'Start yolo_val...'
# python license_plate_det/yolo_val.py

echo 'Start yolo_export...'
python license_plate_det/yolo_export.py

echo 'Completed!'

