"""Transfer Learning from data annotated by the Users in the Web UI"""
import math
import torch
import os
from tqdm.notebook import tqdm
from .SQLDataset import create_dataloader_sql
from torch import optim

from model_dev.helper import load_checkpoint, print_run_config, train_step
from model_dev.augmentations import get_transformation
from model_dev.datasets import get_classes


class Learner:
    def __init__(self,
                 model_path: str,
                 model_output_path: str,
                 batch_size: int,
                 annotation_path: str):
        # Training information
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load model's weight
        weight = model_path
        self.model = load_checkpoint(weight, self.device)
        self.model_path = model_path

        # Get pre-processing steps
        self.transform = get_transformation()

        self.class_list, _ = get_classes(annotation_path)
        self.output_path = model_output_path

    def __model_definition(self):
        # freeze the layers
        for param in self.model.m.features.parameters():
            param.requires_grad = False

        self.model = self.model.to(self.device)

    def train(self,
              epochs: int,
              model_name,
              learning_rate,
              database_path,
              table_name,
              annotation_path,
              strategy="loss"):
        self.__model_definition()

        # Print configuration options
        print_run_config(epochs, strategy, self.output_path, self.model_path, self.model_path, self.device)

        # Create dataset and dataloader
        dataset, dataloader = create_dataloader_sql(database_path,
                                                    table_name,
                                                    self.batch_size,
                                                    annotation_path)

        # Initialize optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training epochs
        min_loss = math.inf
        max_acc = -1
        print("Start training for {} epochs".format(epochs))
        for epoch in tqdm(range(epochs)):
            # Initialize Loss and Accuracy
            train_loss = 0.0
            train_accu = 0.0
            # Iterate over the train dataloader
            with tqdm(total=len(dataloader)) as pbar:
                for idx, sample in enumerate(dataloader):
                    curr_loss, num_correct = train_step(self.model, optimizer, sample)
                    train_loss += curr_loss / len(dataloader)
                    train_accu += num_correct / len(dataset)
                    pbar.update(1)

            # Save the best model based on the training strategy
            checkpoint = {
                'model': self.model,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if self.output_path:
                if strategy == 'loss':
                    min_loss = min(train_loss, min_loss)
                    if min_loss == train_loss:
                        torch.save(checkpoint, self.output_path)
                elif strategy == 'accuracy':
                    max_acc = max(train_accu, max_acc)
                    if max_acc == train_accu:
                        torch.save(checkpoint, self.output_path)

            # Print current epoch, loss, and accuracy
            print("epoch={}".format(epoch))
            print("train_loss={:.3f}, train_accu={:.3f}".format(train_loss, train_accu))


if __name__ == "__main__":
    learner = Learner(model_path="model_best.pth",
                      model_output_path="model_output_tuned.pth",
                      batch_size=2,
                      annotation_path=os.path.join(os.environ.get('ROOT_PATH'), "updated_annotation.json"))
    learner.train(epochs=1,
                  model_name="model_tuned.pth",
                  learning_rate=0.0001,
                  database_path=os.path.join("/mnt", "TUM", "db_labels.db"),
                  table_name="labels",
                  annotation_path=os.path.join(os.environ.get('ROOT_PATH'), "updated_annotation.json"))
