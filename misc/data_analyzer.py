import json
from pathlib import Path
import random
from xml.etree import ElementTree as ET
import cv2
import html
import pandas as pd
import logging


class DataAnalyzer:
    def __init__(self, data_root_path):
        self.data_root_path = Path(data_root_path)

    def parse(self):
        anno_file_paths = list(self.data_root_path.rglob('*.xml'))
        parse_results = []

        for anno_file_path in anno_file_paths:
            # print(anno_file_path.as_posix())

            xml_tree = ET.parse(anno_file_path.as_posix())

            root = xml_tree.getroot()

            self.verify_xml(root, anno_file_path)

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename
            if not image_file_path.exists():
                message = r'Error: {} image {} not exist!'.format(
                anno_file_path,
                image_file_path)
                logging.error(message)
                continue

            size = root.find('size')
            size_dict = self.parse_size(size)

            img = cv2.imread(image_file_path.as_posix())
            image_shape = (size_dict['height'], size_dict['width'], 3)
            message = r'Error: {} image shape {} != {}'.format(
                image_file_path,
                img.shape,
                image_shape)

            if not (img.shape == image_shape):
                logging.error(message)

            obj_list = []

            for object_iter in root.findall('object'):
                name = object_iter.find('name').text
                bnd_box = object_iter.find('bndbox')
                attributes = object_iter.find('attributes')
                polygon = object_iter.find('polygon')

                difficult = object_iter.find('difficult')

                if difficult is not None:
                    is_difficult = bool(int(difficult.text))
                else:
                    is_difficult = False

                bnd_box_dict = self.parse_bnd_box(bnd_box, anno_file_path)
                attribute_dict = self.parse_attributes(attributes, anno_file_path)
                points = self.parse_polygon(polygon, anno_file_path)

                obj = {
                    'name': name,
                    'is_difficult': is_difficult,
                    'bnd_box': bnd_box_dict,
                    'attribute': attribute_dict,
                    'points': points
                }

                obj_list.append(obj)

            parse_result = {
                'anno_file_path': anno_file_path.as_posix(),
                'image_file_path': image_file_path.as_posix(),
                'file_name': filename,
                'size': size_dict,
                'objects': obj_list
            }

            parse_results.append(parse_result)

        return parse_results

    def verify_xml(self, root, anno_file_path):
        child_info_dict = {
            'folder': 1,
            'filename': 1,
            'size': 1,
            'object': 0
        }

        self.verify_node(root,
                         child_info_dict,
                         anno_file_path)

        size = root.find('size')

        child_info_dict = {
            'width': 1,
            'height': 1,
            'depth': 1
        }

        self.verify_node(size,
                         child_info_dict,
                         anno_file_path)

        for object_iter in root.findall('object'):
            child_info_dict = {
                'name': 1,
                'attributes': [0, 1],
                'polygon': [0, 1],
                'bndbox': [0, 1]
            }

            self.verify_node(object_iter,
                             child_info_dict,
                             anno_file_path)

            bnd_box = object_iter.find('bndbox')

            if bnd_box:
                child_info_dict = {
                    'xmin': 1,
                    'ymin': 1,
                    'xmax': 1,
                    'ymax': 1
                }

                self.verify_node(bnd_box,
                                 child_info_dict,
                                 anno_file_path)

            attributes = object_iter.find('attributes')

            if attributes:
                child_info_dict = {
                    'attribute': 2
                }

                self.verify_node(attributes,
                                 child_info_dict,
                                 anno_file_path)

                for attribute in attributes.findall('attribute'):
                    child_info_dict = {
                        'name': 1,
                        'value': 1
                    }

                    self.verify_node(attribute,
                                     child_info_dict,
                                     anno_file_path)

            polygon = object_iter.find('polygon')

            if polygon:
                child_info_dict = {
                    'points': 1
                }

                self.verify_node(polygon,
                                 child_info_dict,
                                 anno_file_path)

    def verify_node(self, node, child_info_dict, anno_file_path):
        for child in node.findall('*'):
            message = r'Error: {} tag {} not in: {}'.format(
                anno_file_path, child.tag, child_info_dict)
            if not (child.tag in child_info_dict):
                logging.error(message)

        for key, count in child_info_dict.items():
            if isinstance(count, int):
                count = [count]

            if len(count) == 1 and count[0] == 0:
                continue

            children = node.findall(key)
            message = r'Error: {} tag {} count {} not in {}'.format(
                anno_file_path, key, len(children), count)

            if not (len(children) in count):
                logging.error(message)

    def parse_size(self, size):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size_dict = {
            'width': width,
            'height': height
        }

        return size_dict

    def parse_bnd_box(self, bnd_box, anno_file_path):
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

    def parse_attributes(self, attributes, anno_file_path):
        if not attributes:
            return None

        attribute_dict = {}

        for attribute in attributes.findall('attribute'):
            name = attribute.find('name').text
            value = attribute.find('value').text
            attribute_dict[name] = value

        if not len(attribute_dict) == 2:
            message = r'Error: {} num_attributes {} != 2'.format(
                anno_file_path,
                attribute_dict)
            logging.error(message)

        keys = ['ocr', 'color']

        for key in keys:
            if not (key in attribute_dict):
                message = r'Error: {} key {} not in {}'.format(
                    anno_file_path,
                    key,
                    attribute_dict)
                logging.error(message)

        if 'ocr' in attribute_dict:
            try:
                attribute_dict['ocr'] = html.unescape(attribute_dict['ocr'])
            except:
                message = r'{} Exception in parsing ocr text: {} attributes: {}'.format(
                    anno_file_path,
                    attribute_dict['ocr'],
                    ET.tostring(attributes, encoding='unicode'))
                logging.exception(message)

        return attribute_dict

    def parse_polygon(self, polygon, anno_file_path):
        if not polygon:
            return None

        points_text = polygon.find('points').text
        point_texts = points_text.split(';')
        points = []

        for point_text in point_texts:
            point_values = point_text.split(',')

            point = [float(point_value)
                for point_value in point_values]

            if not len(point) == 2:
                message = r'Error: {} num_point_value: {} != 2'.format(
                    anno_file_path,
                    point)
                logging.error(message)
            points.append(point)

        # assert len(points) in [4, 6], points
        return points

    def analyze(self, parse_results):
        print('Data files count:', len(parse_results))
        num_image_file_dict = {}
        img_size_dict = {}
        num_gt_dict = {}
        num_class_name_dict = {}
        num_difficult_gt = 0
        num_ocr_text_len_dict = {}
        num_ocr_text_color_dict = {}
        num_points_dict = {}

        for parse_result in parse_results:
            anno_file_path = Path(parse_result['anno_file_path'])
            image_file_path = Path(parse_result['image_file_path'])
            num_image_file_dict.setdefault(image_file_path.suffix, 0)
            num_image_file_dict[image_file_path.suffix] += 1

            size_dict = parse_result['size']
            img_size = (size_dict['width'], size_dict['height'])
            img_size_dict.setdefault(img_size, 0)
            img_size_dict[img_size] += 1

            obj_list = parse_result['objects']
            num_gt = len(obj_list)
            num_gt_dict.setdefault(num_gt, 0)
            num_gt_dict[num_gt] += 1

            for obj in obj_list:
                class_name = obj['name']
                num_class_name_dict.setdefault(class_name, 0)
                num_class_name_dict[class_name] += 1

                is_difficult = obj['is_difficult']
                if is_difficult:
                    num_difficult_gt += 1

                bnd_box_dict = obj['bnd_box']
                attribute_dict = obj['attribute']
                points = obj['points']

                if bnd_box_dict:
                    message = r'Error: one of {} attribute_dict {} points {} is not empty, bnd_box {}'.format(
                        anno_file_path, attribute_dict, points, bnd_box_dict)

                    if (attribute_dict or points):
                        logging.error(message)

                if attribute_dict or points:
                    message = r'Error: {} attribute_dict {} points {}, bnd_box {} is not empty'.format(
                        anno_file_path, attribute_dict, points, bnd_box_dict)

                    if bnd_box_dict:
                        logging.error(message)

                    message = r'Error: {} one of attribute_dict {} points {} is empty!'.format(
                        anno_file_path, attribute_dict, points)
                    if not (attribute_dict and points):
                        logging.error(message)

                    gt_ocr_text = attribute_dict['ocr']

                    if gt_ocr_text is None:
                        continue

                    ocr_text_len = len(gt_ocr_text)
                    num_ocr_text_len_dict.setdefault(ocr_text_len, 0)
                    num_ocr_text_len_dict[ocr_text_len] += 1

                    color = attribute_dict['color']
                    num_ocr_text_color_dict.setdefault(color, 0)
                    num_ocr_text_color_dict[color] += 1

                    num_points = len(points)
                    num_points_dict.setdefault(num_points, 0)
                    num_points_dict[num_points] += 1

        self.print_numbers('num_image_file_dict', num_image_file_dict)
        self.print_numbers('img_size_dict', img_size_dict)
        self.print_numbers('num_gt_dict', num_gt_dict)
        self.print_numbers('num_class_name_dict', num_class_name_dict)
        self.print_numbers('num_difficult_gt', num_difficult_gt)
        self.print_numbers('num_ocr_text_len_dict', num_ocr_text_len_dict)
        self.print_numbers('num_ocr_text_color_dict', num_ocr_text_color_dict)
        self.print_numbers('num_points_dict', num_points_dict)

    def print_numbers(self, comment, numbers):
        print('#' * 80)
        print(comment)
        print('-' * 80)
        print(numbers)
        print('#' * 80)

def main():
    print()
    print('=' * 80)
    data_root_path = r'/home/data'
    data_analyzer = DataAnalyzer(data_root_path)
    parse_results = data_analyzer.parse()

    data_analyzer.analyze(parse_results)
    print('=' * 80)
    print()


if __name__ == '__main__':
    main()
