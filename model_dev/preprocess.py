import os
import json
import shutil
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm

import cv2
from pycocotools.coco import COCO


def preprocess_data(data_path, label_path, single_label_file, out_path):
    """
    Pre-process data by updating the annotation file with the new 4 labels and saving the damage patches
    :param data_path: path to the raw image data
    :param label_path: path to the labels
    :param single_label_file:
    :param out_path: path to save the damage patches
    :return:
    """
    # Get images and annotation path
    img_path = os.path.join(data_path, f'Images')
    annotation_path = os.path.join(data_path, f'Annotations/annotated_functional_test3_fixed.json')
    updated_ann_path = "updated_annotation.json"

    # Update annotations
    json_obj = JSONLabelFile(annotation_path, label_path, single_label_file, save_path=updated_ann_path)
    json_obj.execute_all_tasks()

    # Save the cropped damage patch from the annotated images
    ann_img_path = os.path.join(out_path, f'new_dataset')
    img_label_obj = ImagesAndLabels(updated_ann_path, label_path, img_path, ann_img_path)
    img_label_obj.save_images()

    return ann_img_path, updated_ann_path


def clean_up_labels(labels_path):
    """
    Automatic function to resolve conflicting labels after manual annotation
    :param labels_path: directory containing the conflicting labels
    :return:
    """
    data = {}
    for label_name in os.listdir(labels_path):
        filepath = os.path.join(labels_path, label_name)
        data[label_name[7:-4]] = pd.read_csv(filepath, sep=",", header=0)

    list_label_name = [data[name] for name, label in data.items()]
    label_complete = pd.concat(list_label_name).drop_duplicates(subset=[" ID"], keep='first')

    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(this_dir)
    save_path = os.path.join(root_dir, r'label_complete.txt',)
    label_complete.to_csv(save_path, header=None, index=None, sep=',')

    return label_complete


class JSONLabelFile:
    def __init__(self, json_path, label_path, single_label_file=True, save_path="updated_annotation.json"):
        self.json_path = json_path
        self.json_file = json.load(open(json_path))
        self.single_label_file = single_label_file
        self.label_path = label_path
        self.save_path = save_path

    def define_categories(self):
        """For coco format, we need to define the categories aka object classes
        """
        scratch_dict = {"id": 1, "name": "scratch"}
        dent_dict = {"id": 2, "name": "dent"}
        rim_dict = {"id": 3, "name": "rim"}
        other_dict = {"id": 4, "name": "other"}

        # In COCO format, categories is a list of dictionaries. The dictionaries are the object classes.
        self.json_file['categories'] = [scratch_dict, dent_dict, rim_dict, other_dict]

    def check_single_label_file(self):
        """If single_label_file is true, the given label path needs to be a path to a file. Otherwise to a directory.

        Raises:
            Exception: Exception is raised if single_label_file is true and the given label_path is not a file.
            Exception: Exception is raised if single_label_file is false and the given label_path is not a directory.
        """
        label_path_obj = Path(self.label_path)
        if self.single_label_file:
            if not label_path_obj.is_file():
                raise Exception('the label_path should point to a SINGLE txt file containing labels!')
        else:
            if not label_path_obj.is_dir():
                raise Exception('the label_path should point to a DIRECTORY containing txt files!')

    def update_categories(self, label_path):
        """Update the categories for the image_ids listed in the label file

        Args:
            label_path (str): path to a txt file containing labels
        """
        open_label_file = open(label_path, 'r')
        lines = open_label_file.read().splitlines()
        for line in lines:
            try:
                im_id, id_id, cat_id = line.split(",")

                # In COCO format, annotations is a list of dictionaries.
                # The dictionaries contain the image_id and category_id.
                for annotation in self.json_file['annotations']:
                    if annotation['image_id'] == int(im_id):
                        if annotation['id'] == int(id_id):
                            annotation['category_id'] = int(cat_id)
            except:
                if line == '' or line == '\n':
                    print("detected extra line at the end of file")
                else:
                    print(f"cannot parse this line: {line}")
                break

    def update_json_file(self):
        """update json file either with one single label file, or multiple label files in a dir.
        """
        if self.single_label_file:
            self.update_categories(self.label_path)
        else:
            label_paths = Path(self.label_path).glob('*.txt')
            for label_path in label_paths:
                self.update_categories(label_path)

    def save_updated_json(self):
        """save the updated json into a new json file
        """
        with open(self.save_path, 'w') as file:
            json.dump(self.json_file, file)

    def execute_all_tasks(self):
        """execute all necessary steps to update the categories for annotated images.
        """
        self.define_categories()
        self.check_single_label_file()
        self.update_json_file()
        self.save_updated_json()


class ImagesAndLabels:
    def __init__(self, annotation_path, label_path, img_path, save_path):
        self.annotation_path = annotation_path
        self.label_path = label_path
        self.img_path = img_path
        self.save_path = save_path

    def save_images(self):
        """
        Save cropped damage patch
        :return:
        """
        # Create save folder, overwrite old files if save folder already exists
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)

        # Load coco annotation
        coco = COCO(self.annotation_path)
        label_file = open(self.label_path, 'r')
        lines = label_file.read().splitlines()
        print('--------------------------------')
        print("Number of labeled images: {}".format(len(lines)))
        print("Loading labeled images")
        print("Saving cropped damage patches in {}".format(self.save_path))
        with tqdm(total=len(lines)) as pbar:
            for line in lines:
                # Load image
                img_id, _, _ = line.split(",")
                img = coco.loadImgs(int(img_id))[0]
                file_name = img['file_name']
                img_path = os.path.join(self.img_path, f'{file_name}')
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)

                # Get the damaged area of the image
                ann_id = coco.getAnnIds(imgIds=int(img_id))
                ann = coco.loadAnns(ann_id)
                pos = ann[0]['segmentation'][0]
                img_seg = img[pos[1]:pos[7], pos[0]:pos[2], :]

                # Save the cropped damage patch
                cv2.imwrite(os.path.join(self.save_path, img_name.replace(".webp", ".jpeg")), img_seg)
                pbar.update(1)
