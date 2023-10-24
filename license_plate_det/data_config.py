import json
from pathlib import Path
import random
import cv2
import pandas as pd
from xml.etree import ElementTree as ET
import logging
import html
import yaml
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw


def set_logging(log_file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        # format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        handlers=[
            logging.FileHandler(log_file_path.as_posix(), mode='a'),
            logging.StreamHandler()
        ])


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class DataConfigManager:
    def __init__(self, config_file_path_dict):
        self.config_file_path_dict = config_file_path_dict

    def generate(self):
        logging.info('Start parse_anno_info...')
        anno_info_list = self.parse_anno_info()
        self.analyze_anno_infos(anno_info_list)

        logging.info('Start generate_yolo_configs...')
        # self.generate_yolo_configs(anno_info_list)

    def parse_anno_info(self):
        anno_info_list = []

        data_root_path = self.config_file_path_dict['path']
        anno_file_paths = list(data_root_path.rglob('*.xml'))

        for anno_file_path in anno_file_paths:
            xml_tree = ET.parse(anno_file_path.as_posix())
            root = xml_tree.getroot()

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename
            size = root.find('size')
            size_dict = self.parse_size(size)

            obj_list = []

            for object_iter in root.findall('object'):
                name = object_iter.find('name').text
                bnd_box = object_iter.find('bndbox')
                attributes = object_iter.find('attributes')
                polygon = object_iter.find('polygon')
                difficult = object_iter.find('difficult')

                is_difficult = self.parse_difficult(difficult)
                bnd_box_dict = self.parse_bnd_box(bnd_box)
                attribute_dict = self.parse_attributes(attributes)
                points = self.parse_polygon(polygon)

                obj = {
                    'name': name,
                    'is_difficult': is_difficult,
                    'bnd_box': bnd_box_dict,
                    'attribute': attribute_dict,
                    'points': points
                }

                obj_list.append(obj)

            anno_info = {
                'anno_file_path': anno_file_path,
                'image_file_path': image_file_path,
                'size': size_dict,
                'objects': obj_list
            }

            anno_info_list.append(anno_info)

        return anno_info_list

    def parse_size(self, size):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size_dict = {
            'width': width,
            'height': height
        }

        return size_dict

    def parse_bnd_box(self, bnd_box):
        if not bnd_box:
            return None

        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        bnd_box_dict = {
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max
        }

        return bnd_box_dict

    def parse_difficult(self, difficult):
        if difficult is None:
            return None

        is_difficult = bool(int(difficult.text))
        return is_difficult

    def parse_attributes(self, attributes):
        if not attributes:
            return None

        attribute_dict = {}

        for attribute in attributes.findall('attribute'):
            name = attribute.find('name').text
            value = attribute.find('value').text
            attribute_dict[name] = value

        if not len(attribute_dict) == 2:
            message = r'Error: num_attributes {} != 2'.format(
                attribute_dict)
            logging.error(message)

        keys = ['ocr', 'color']

        for key in keys:
            if not (key in attribute_dict):
                message = r'Error: key {} not in {}'.format(
                    key,
                    attribute_dict)
                logging.error(message)

        if 'ocr' in attribute_dict and (attribute_dict['ocr'] is not None):
            try:
                attribute_dict['ocr'] = html.unescape(attribute_dict['ocr'])
            except:
                message = r'{Exception in parsing ocr text: {} attributes: {}'.format(
                    attribute_dict['ocr'],
                    ET.tostring(attributes, encoding='unicode'))
                logging.exception(message)

        return attribute_dict

    def parse_polygon(self, polygon):
        if not polygon:
            return None

        points_text = polygon.find('points').text
        point_texts = points_text.split(';')
        points = []

        for point_text in point_texts:
            point_values = point_text.split(',')

            point = [float(point_value)
                for point_value in point_values]

            if len(point) != 2:
                message = r'Error: num_point_value: {} != 2'.format(
                    point)
                logging.error(message)
            points.append(point)

        if len(points) not in [4, 6]:
            message = r'Error: points len: {} not in [4, 6]'.format(
                points)
            logging.error(message)

        return points

    def analyze_anno_infos(self,
        anno_info_list,
        is_show_obj_size=False,
        is_visualize=False):
        model_input_size = 640
        label_color = (255, 0, 0) # Red
        label_fill_color = (0, 255, 0) # Green

        class_name_dict = {}
        num_gt_dict = {}

        for anno_info in anno_info_list:
            anno_file_path = anno_info['anno_file_path']
            image_file_path = anno_info['image_file_path']
            size_dict = anno_info['size']
            obj_list = anno_info['objects']

            num_gt = 0

            resize_ratio_dict = {
                'width': model_input_size / size_dict['width'],
                'height': model_input_size / size_dict['height']
            }

            if is_visualize:
                image = Image.open(image_file_path.as_posix())
                draw = ImageDraw.Draw(image, 'RGBA')

            for obj in obj_list:
                name = obj['name']
                is_difficult = obj['is_difficult']
                bnd_box_dict = obj['bnd_box']
                attribute_dict = obj['attribute']
                points = obj['points']

                if name != 'plate':
                    continue

                ocr_text = attribute_dict['ocr']
                ocr_color = attribute_dict['color']

                class_name_dict.setdeault(ocr_color, 0)
                class_name_dict[ocr_color] += 1

                num_gt += 1

                if is_show_obj_size:
                    points_str = ' '.join([' '.join([str(coordinate) for coordinate in point]) for point in points])

                    x_min = min([point[0] for point in points])
                    x_max = max([point[0] for point in points])
                    y_min = min([point[1] for point in points])
                    y_max = max([point[1] for point in points])

                    resized_obj_size = {
                        'width': (x_max - x_min) * resize_ratio_dict['width'],
                        'height': (y_max - y_min) * resize_ratio_dict['height']
                    }

                    print(image_file_path,
                        ocr_text,
                        ocr_color,
                        resized_obj_size['width'],
                        resized_obj_size['height'],
                        points_str)

                if is_visualize:
                    fill_color = (*label_fill_color, 127)
                    draw.polygon([tuple(point) for point in points],
                                 fill=fill_color,
                                 outline=label_color)

            num_gt_dict.setdeault(num_gt, 0)
            num_gt_dict[num_gt] += 1

            image_file_name = r'{}_paint{}'.format(image_file_path.stem, image_file_path.suffix)
            paint_image_file_path = image_file_path.parent.parent / 'paint' / image_file_name
            paint_image_file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(paint_image_file_path.as_posix())

        logging.info(r'num_gt_dict: {}'.format(num_gt_dict))

    def generate_yolo_configs(self,
                              anno_info_list,
                              max_num_val_data=1000,
                              max_val_percent=0.2,
                              seed=7,
                              expand_data=False):
        config_file_path_dict = self.config_file_path_dict
        class_name_dict = {}

        for anno_info in anno_info_list:
            anno_file_path = anno_info['anno_file_path']
            image_file_path = anno_info['image_file_path']
            size_dict = anno_info['size']
            obj_list = anno_info['objects']
            num_gt = 0

            for obj in obj_list:
                name = obj['name']
                is_difficult = obj['is_difficult']
                bnd_box_dict = obj['bnd_box']
                attribute_dict = obj['attribute']
                points = obj['points']

                if name != 'plate':
                    continue

                ocr_text = attribute_dict['ocr']
                ocr_color = attribute_dict['color']

                class_name_dict.setdeault(ocr_color, 0)
                class_name_dict[ocr_color] += 1

        logging.info(r'class_name_dict: {}'.format(num_gt_dict))
        class_names = list(class_name_dict.keys())

        for anno_info in anno_info_list:
            anno_file_path = anno_info['anno_file_path']
            image_file_path = anno_info['image_file_path']
            size_dict = anno_info['size']
            obj_list = anno_info['objects']

            anno_contents = []

            for obj in obj_list:
                name = obj['name']
                is_difficult = obj['is_difficult']
                bnd_box_dict = obj['bnd_box']
                attribute_dict = obj['attribute']
                points = obj['points']

                if name != 'plate':
                    continue

                ocr_text = attribute_dict['ocr']
                ocr_color = attribute_dict['color']

                class_index = class_names.index(ocr_color)
                normed_coordinates = []

                for x, y in points:
                    normed_x = x / size_dict['width']
                    normed_y = y / size_dict['width']
                    normed_coordinates.append(normed_x)
                    normed_coordinates.append(normed_y)

                normed_coordinates_str = ' '.join(
                    [str(normed_coordinate) for normed_coordinate in normed_coordinates])
                line = '{} {}'.format(
                    class_index,
                    normed_coordinates_str)
                anno_contents.append(line)

            size_dict = anno_info['size']
            image_width = size_dict['width']
            image_height = size_dict['height']

            image_file_path = Path(anno_info['image_file_path'])
            anno_config_file_path = image_file_path.with_suffix('.txt')

            with open(anno_config_file_path, 'w') as file_stream:
                for line in anno_contents:
                    file_stream.write('{}\n'.format(line))

        set_random_seed(seed)
        random.shuffle(anno_info_list)

        num_val_data = min(max_num_val_data,
                           int(len(anno_info_list) * max_val_percent))

        anno_infos_dict = {
            'train': anno_info_list[:-num_val_data],
            'val': anno_info_list[-num_val_data:]
        }

        for data_type, anno_infos in anno_infos_dict.items():
            message = r'{}: writing file {} with num_data {}'.format(
                data_type,
                config_file_path_dict[data_type],
                len(anno_infos))
            logging.info(message)

            with open(config_file_path_dict[data_type], 'w') as file_stream:
                for anno_info in anno_infos:
                    image_file_path = anno_info['image_file_path']
                    file_stream.write('{}\n'.format(image_file_path))

        dataset_config = {
            'path': config_file_path_dict['path'].as_posix(),
            'train': config_file_path_dict['train'].name,
            'val': config_file_path_dict['val'].name,
            'names': {
                class_index: class_name
                for class_index, class_name
                in enumerate(class_names)
            }
        }

        message = r'Writing dataset config file: {}'.format(
            config_file_path_dict['dataset'])
        logging.info(message)

        with open(config_file_path_dict['dataset'], 'w') as file_stream:
            yaml.dump(dataset_config, file_stream, indent=4)


def main():
    data_root_path = Path(r'/home/data')

    config_file_path_dict = {
        'path': data_root_path,
        'train': data_root_path / 'train.txt',
        'val': data_root_path / 'val.txt',
        'dataset': data_root_path / 'custom_dataset.yaml',
        'dataset_info': data_root_path / 'dataset_info.json'
    }

    log_file_path = Path('/project/train/log/log.txt')

    set_logging(log_file_path)

    logging.info('=' * 80)
    logging.info('Start DataConfigManager')
    data_manager = DataConfigManager(config_file_path_dict)
    data_manager.generate()
    logging.info('End DataConfigManager')
    logging.info('=' * 80)


if __name__ == '__main__':
    main()
