import json
from pathlib import Path
import re
import logging
from xml.etree import ElementTree as ET
import cv2
import sys

sys.path.append('/project/train/src_repo/ultralytics')

from ultralytics.utils import LOGGER as logger
from ultralytics import YOLO


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
        model_file_path = child_path  / 'weights' / 'best.pt'

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


def init():
    log_file_path = r'/project/train/log/log.txt'
    set_logging(log_file_path)

    model_file_path = find_model_file_path()
    logger.info(r'model_file_path: {}'.format(model_file_path))

    model = YOLO(model_file_path.as_posix())
    return model


def main():
    model = init()
    model.export(format='engine', simplify=True, half=True)
    del model

if __name__ == '__main__':
    main()
