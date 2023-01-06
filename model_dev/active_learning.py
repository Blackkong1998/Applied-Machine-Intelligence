import os
import json
import glob
import shutil
from pathlib import Path
from PIL import Image
from functools import partial

import torch
from sklearn.model_selection import train_test_split
from .datasets import get_classes

from baal.active import get_heuristic
from flash import Trainer
from flash.image import ImageClassifier, ImageClassificationData
from flash.core.classification import LogitsOutput
from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
    ActiveLearningLoop,
)


def prepare_folder(data_path, new_root_path, annotation_path):
    """
    Prepare folders to start active learning
    :param data_path: path to original data
    :param new_root_path: path to organized data for active learning
    :param annotation_path: path to the annotation file
    :return:
    """
    if os.path.exists(new_root_path):
        shutil.rmtree(new_root_path)
    os.makedirs(new_root_path, exist_ok=True)

    # create folder for each class
    class_list, _ = get_classes(annotation_path)
    for c in class_list:
        c_path = new_root_path + c + '/'
        os.makedirs(c_path, exist_ok=True)

    json_data = json.load(open(annotation_path))
    for file_path in data_path:
        for im in json_data['images']:
            if Path(im['file_name']).stem == Path(file_path).stem:
                for ann in json_data['annotations']:
                    if im['id'] == ann['image_id']:
                        class_name = class_list[ann['category_id'] - 1]
                        new_path = new_root_path + class_name + '/' + Path(file_path).stem
                        with Image.open(file_path) as img:
                            img.save(new_path + ".jpeg", "JPEG")


def get_data_module(train, test, heuristic_name):
    """
    Return a data module for active learning
    :param train: train data
    :param test: test data
    :param heuristic_name: active learning heuristic
    :return:
    """
    datamodule = ImageClassificationData.from_folders(
        train_folder=train,
        test_folder=test,
        batch_size=16,
    )

    active_datamodule = ActiveLearningDataModule(
        datamodule,
        heuristic=get_heuristic(heuristic_name),
        initial_num_labels=100,
        query_size=128,
        val_split=0.0,
    )

    return active_datamodule


def get_model(datamodule, feat_dim=512, backbone="resnet18"):
    """
    Return a model to be tested for active learning
    :param datamodule: data module
    :param feat_dim: dimension of the last layer
    :param backbone: backbone model
    :return:
    """
    head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(feat_dim, datamodule.num_classes),
    )

    model = ImageClassifier(
        num_classes=4,
        backbone=backbone,
        head=head,
        pretrained=True,
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
    )
    model.serializer = LogitsOutput()

    return model


def start_active_learning(model, active_datamodule):
    """
    Start active learning
    :param model: model to be tested
    :param active_datamodule: data module for active learning
    :return:
    """
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(
        gpus=gpus,
        max_epochs=5,
    )

    # Train for 20 epochs before doing 20 MC-Dropout iterations to estimate uncertainty.
    active_learning_loop = ActiveLearningLoop(
        label_epoch_frequency=20,
        inference_iteration=20
    )
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop
    trainer.finetune(model, datamodule=active_datamodule, strategy="no_freeze")


if __name__ == "__main__":
    input_path = '../../../Data/new_dataset/Images'
    updated_ann_path = '../../../Data/new_dataset/Annotations/updated_annotation.json'
    train_folder = '/home/mikhaeldj/TUM/AMI/active_learning_data/train/'
    test_folder = '/home/mikhaeldj/TUM/AMI/active_learning_data/test/'

    input_data = glob.glob(os.path.join(os.path.abspath(input_path), "*.jpeg"))
    train_data, test_data = train_test_split(input_data, test_size=0.2, random_state=0)
    # prepare_folder(train_data, train_folder, updated_ann_path)
    # prepare_folder(test_data, test_folder, updated_ann_path)

    heuristic = "bald"
    model_backbone = "resnet18"
    feat_size = 512

    dm = get_data_module(train_folder, test_folder, heuristic)
    m = get_model(dm, feat_size, model_backbone)
    start_active_learning(m, dm)
