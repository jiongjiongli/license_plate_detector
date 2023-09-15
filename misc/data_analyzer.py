import json
from pathlib import Path
import random
from xml.etree import ElementTree as ET


class DataAnalyzer:
    def __init__(self):
        self.data_root_path = Path(r'/home/data')

    def parse(self):
        anno_file_paths = list(self.data_root_path.rglob('*.xml'))

        print('=' * 80)
        print('Data files count:', len(anno_file_paths))

        ann_file_index = random.randint(0, len(anno_file_paths))
        anno_file_paths = [anno_file_paths[ann_file_index]]

        print('Parsing data file at: {}'.format(ann_file_index))

        for anno_file_path in anno_file_paths:
            print(anno_file_path.as_posix())

            xml_tree = ET.parse(anno_file_path.as_posix())

            print(ET.tostring(xml_tree))

            # root = xml_tree.getroot()
            # filename = root.find('filename').text

            # size = root.find('size')
            # size_dict = self.parse_size(size)
            # box_list = []

            # for object_iter in root.findall('object'):
            #     bnd_box = object_iter.find('bndbox')
            #     name = object_iter.find('name').text

            #     difficult = False

            #     if object_iter.find('difficult') is not None:
            #         difficult = bool(int(object_iter.find('difficult').text))

            #     bnd_box_dict = self.parse_bnd_box(name, bnd_box, difficult)
            #     box_list.append(bnd_box_dict)

            # parse_result = {'file_name': filename,
            #                 'size': size_dict,
            #                 'box_list': box_list}

        print('=' * 80)

    def parse_size(self, size):
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        depth = int(size.find('depth').text)

        size_dict = {'height': height, 'width': width, 'depth': depth}
        return size_dict


def main():
    data_analyzer = DataAnalyzer()
    data_analyzer.parse()


if __name__ == '__main__':
    main()
