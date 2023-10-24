import json
from pathlib import Path
import re
import logging
from xml.etree import ElementTree as ET
import cv2
import numpy as np
import sys

sys.path.append('/project/train/src_repo/ultralytics')

from ultralytics.utils import LOGGER as logger
from ultralytics import YOLO
from vehicle_det.yolo_export import main as export_main


def set_logging(log_file_path):
    file_handler = logging.FileHandler(Path(log_file_path).as_posix())
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def find_model_file_path():
    model_save_dir_path = Path('/project/train/models')
    child_paths = list(model_save_dir_path.glob('train*'))

    model_file_infos = []

    for child_path in child_paths:
        if not child_path.is_dir():
            continue

        dir_name = child_path.name
        model_file_path = child_path  / 'weights' / 'best.engine'

        if not model_file_path.exists():
            continue

        model_file_info = {
            'model_file_path': model_file_path,
            'dir_name': dir_name
        }

        model_file_infos.append(model_file_info)

    assert model_file_infos, r'No valid model_file_path!'

    best_model_file_path = None
    best_dir_index = -1

    for model_file_info in model_file_infos:
        dir_name = model_file_info['dir_name']
        match_results = re.match(r'^train([0-9]+$)', dir_name)

        if match_results:
            dir_index = int(match_results.group(1))
        else:
            dir_index = 0

        if dir_index > best_dir_index:
            best_dir_index = dir_index
            best_model_file_path = model_file_info['model_file_path']

    return best_model_file_path


def process_image(model, input_image=None, args=None, **kwargs):
    fake_result = {
        'algorithm_data': {
            'is_alert': False,
            'target_count': 0,
            'target_info': []
        },
        'model_data': {'objects': []}
    }

    conf_thresh = 0.8
    iou_thresh = 0.7
    results = model(input_image, conf=conf_thresh,
                    iou=iou_thresh, half=True)

    object_infos = []

    for result in results:
        class_names = result.names
        boxes = result.boxes
        xyxy_tensor = boxes.xyxy
        conf_tensor = boxes.conf
        cls_tensor  = boxes.cls

        for xyxy, conf, class_index in zip(xyxy_tensor, conf_tensor, cls_tensor):
            object_info = {
                'x':int(xyxy[0]),
                'y':int(xyxy[1]),
                'width':int(xyxy[2]-xyxy[0]),
                'height':int(xyxy[3]-xyxy[1]),
                'confidence':float(conf),
                'name':class_names[int(class_index)]
            }

            object_infos.append(object_info)

    fake_result['model_data']['objects'] = object_infos

    result_str = json.dumps(fake_result, indent = 4)
    return result_str


def init():
    export_main()

    log_file_path = r'/project/train/log/log.txt'
    set_logging(log_file_path)

    model_file_path = find_model_file_path()
    logger.info(r'model_file_path: {}'.format(model_file_path))

    model = YOLO(model_file_path.as_posix())

    random_img = np.random.randint(0,
                                   256,
                                   size=(512, 512, 3),
                                   dtype=np.uint8)
    process_image(model, random_img)
    return model


def main():
    data_root_path = Path(r'/home/data')

    anno_file_paths = list(data_root_path.rglob('*.xml'))
    anno_file_paths = anno_file_paths[:2]

    model = init()

    for anno_file_path in anno_file_paths:
        xml_tree = ET.parse(anno_file_path.as_posix())
        root = xml_tree.getroot()

        filename = root.find('filename').text
        image_file_path = anno_file_path.parent / filename
        img = cv2.imread(image_file_path.as_posix())
        result_str = process_image(model, img)

        logger.info(r'image_file_path: {}'.format(image_file_path))

        logger.info(r'result_str: {}'.format(result_str))

if __name__ == '__main__':
    main()
