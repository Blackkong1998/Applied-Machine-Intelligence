import os
import json

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from .augmentations import get_transformation


def create_dataloader(inputs, annotation_path, batch_size, shuffle=True):
    """
    Create dataset and dataloader for PyTorch model
    :param inputs: input data
    :param annotation_path: path to the annotation file
    :param batch_size: batch size
    :param shuffle: bool to shuffle the dataloader
    :return:
    """
    dataset = WennDataset(inputs, annotation_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, dataloader


def get_classes(annotation_path):
    """
    Get class labels and number of labels
    :param annotation_path: path to the annotation file
    :return:
    """
    ann_file = json.load(open(annotation_path))
    categories = ann_file['categories']
    num_classes = len(categories)
    class_list = list()
    for i in range(num_classes):
        cat = categories[i]
        class_name = cat['name']
        class_list.append(class_name)

    return class_list, num_classes


def visualize_sample(inputs, annotation_path, batch_size, shuffle=True):
    """
    Load some samples and plot them
    :param inputs: input data
    :param annotation_path: path to the annotation file
    :param batch_size: batch size
    :param shuffle: bool to shuffle the dataloader
    :return:
    """
    _, dataloader = create_dataloader(inputs, annotation_path, batch_size, shuffle=shuffle)

    # Take some samples
    sample = next(iter(dataloader))
    print("Input shape: {}".format(sample['input'].shape))

    # Plot images
    fig, ax = plt.subplots(1, 7, figsize=(20, 10))
    for i in range(7):
        ax[i].imshow(sample['input'][i].permute(1, 2, 0))
        ax[i].set_title(sample['class_name'][i])


class WennDataset(Dataset):
    def __init__(self, data_paths, annotation_path):
        self.data_paths = data_paths
        self.class_list, _ = get_classes(annotation_path)

        # Add transformations
        self.transform = get_transformation()

        # Information for __len__ and __getitem__
        self.path_and_label = list()
        json_data = json.load(open(annotation_path))
        for file_path in self.data_paths:
            for im in json_data['images']:
                im_file_name = os.path.splitext(os.path.basename(im['file_name']))[0]
                if im_file_name == os.path.splitext(os.path.basename(file_path))[0]:
                    for ann in json_data['annotations']:
                        if im['id'] == ann['image_id']:
                            cat_id = ann['category_id']
                            cat_name = self.class_list[cat_id - 1]
                            self.path_and_label.append((file_path, cat_name, cat_id))

    def __len__(self):
        return len(self.path_and_label)

    def __getitem__(self, idx):
        (img_path, class_name, class_label) = self.path_and_label[idx]
        img = Image.open(img_path)
        inp = self.transform(img)

        sample = dict()
        sample['input'] = inp
        sample['class_name'] = class_name
        sample['label'] = class_label-1

        return sample
