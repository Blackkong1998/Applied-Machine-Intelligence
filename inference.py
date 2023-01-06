import os
from typing import Union

from PIL import Image
import torch
from torch.nn.functional import softmax
from werkzeug.datastructures import FileStorage

from model_dev.augmentations import get_transformation
from model_dev.datasets import get_classes
from model_dev.helper import load_checkpoint


class TrainedModel:
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.this_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        annotation_path = os.path.join(self.this_dir, "new_dataset/Annotations/updated_annotation.json")
        self.class_list, _ = get_classes(annotation_path)

        # Load model's weight
        weight = os.path.join(self.this_dir, "model_best.pth")
        self.model = load_checkpoint(weight, self.device)
        self.model.eval()

        # Get pre-processing steps
        self.transform = get_transformation()

    def predict(self, input_img: Union[str, FileStorage]):
        """
        Predict using the trained model
        :param input_img: input to the model to be classified
        :return:
        """
        # Open image and input to model
        img = Image.open(input_img).convert('RGB')
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)
        out = self.model(batch_t)  # Get model's output

        # Decode model's output to get predicted label and corresponding confidence
        _, index = torch.max(out, 1)
        probabilities = softmax(out, dim=1)[0]
        pred_label = self.class_list[index[0]]
        pred_prob = probabilities[index[0]].item()

        return pred_label, pred_prob

