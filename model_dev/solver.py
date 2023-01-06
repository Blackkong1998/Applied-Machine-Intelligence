import os
import math
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from .datasets import create_dataloader
from .model import Model
from .helper import train_step, run_train_batch, run_test_batch, reset_weight, print_run_config, load_checkpoint


def train(input_path,
          annotation_path,
          out_path,
          epochs,
          batch_size,
          learning_rate=1e-4,
          loader_shuffle=True,
          model_name='model_best.pth',
          save_model=True,
          saving_strategy='loss',
          device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    """
    Train a deep learning model
    :param input_path: path to the train dataset
    :param annotation_path: path to the annotation file
    :param out_path: path to saving the trained model
    :param epochs: number of training epochs
    :param batch_size: batch size
    :param learning_rate: learning rate for the optimizer
    :param loader_shuffle: bool to shuffle the dataloader
    :param model_name: name of the model to be saved
    :param save_model: bool to save the model
    :param saving_strategy: strategy to save the model ("loss" or "accuracy")
    :param device: device to be used (GPU or CPU)
    :return:
    """
    # Create folder for saving the model
    model_dir = os.path.join(out_path, f'models')
    os.makedirs(model_dir, exist_ok=True)

    # Print configuration options
    print_run_config(epochs, saving_strategy, save_model, model_dir, model_name, device)

    # Create dataset and dataloader
    dataset, dataloader = create_dataloader(input_path, annotation_path, batch_size, shuffle=loader_shuffle)

    # Initialize model
    model = Model()
    model = model.to(device)
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

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
                curr_loss, num_correct = train_step(model, optimizer, sample)
                train_loss += curr_loss / len(dataloader)
                train_accu += num_correct / len(dataset)
                pbar.update(1)

        # Save the best model based on the training strategy
        checkpoint = {
            'model': Model(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if save_model:
            if saving_strategy == 'loss':
                min_loss = min(train_loss, min_loss)
                if min_loss == train_loss:
                    torch.save(checkpoint, os.path.join(model_dir, model_name))
            elif saving_strategy == 'accuracy':
                max_acc = max(train_accu, max_acc)
                if max_acc == train_accu:
                    torch.save(checkpoint, os.path.join(model_dir, model_name))

        # Print current epoch, loss, and accuracy
        print("epoch={}".format(epoch))
        print("train_loss={:.3f}, train_accu={:.3f}".format(train_loss, train_accu))


def test(test_path,
         annotation_path,
         model_path,
         batch_size=16,
         loader_shuffle=False,
         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    """
    Test a trained model
    :param test_path: path to the test dataset
    :param annotation_path: path to the annotation file
    :param model_path: path to the trained model to be tested
    :param batch_size: batch size
    :param loader_shuffle: bool to shuffle the dataloader
    :param device: device to be used (GPU or CPU)
    :return:
    """
    # Load model
    model = load_checkpoint(model_path, device)

    # Create dataset and dataloader
    dataset, dataloader = create_dataloader(test_path, annotation_path, batch_size, shuffle=loader_shuffle)

    # Run test batch
    _, test_accu, recall, y_pred, y_true = run_test_batch(model, dataloader)

    print('Test Accuracy: ', test_accu)
    print('Test Recall: ', recall)

    return y_pred, y_true


class Validator:
    def __init__(self, input_path, annotation_path, out_path,
                 epochs, batch_size, learning_rate=1e-4, num_folds=5,
                 model_name='model_best.pth', save_model=False, saving_strategy='loss'):
        # Paths
        self.input_path = input_path
        self.annotation_path = annotation_path
        self.out_path = out_path

        # Hyper-parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_folds = num_folds

        # Settings for saving model
        self.model_name = model_name
        self.save_model = save_model
        self.saving_strategy = saving_strategy

        # Fixed setting
        self.tmp_path = './checkpoint.pth'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Output
        self.results = {}

    def stratified_k_fold(self):
        """
        Run stratified k-fold cross validation
        :return:
        """
        # Create folder for saving the model
        model_dir = os.path.join(self.out_path, f'models')
        os.makedirs(model_dir, exist_ok=True)

        # Print configuration options
        print_run_config(self.epochs, self.saving_strategy, self.save_model, model_dir, self.model_name, self.device)

        # Create dataset and dataloader
        dataset, _ = create_dataloader(self.input_path, self.annotation_path, self.batch_size, shuffle=True)
        # List of target classes
        target_list = list()
        for i in range(len(dataset)):
            target = dataset[i]['class_name']
            target_list.append(target)

        # Define K-fold Cross Validator
        k_fold = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        # Dict to save results
        fold_accu = {}
        fold_recall = {}
        # Stratified K-fold Cross Validation model evaluation
        for fold, (train_id, val_id) in enumerate(k_fold.split(dataset, target_list)):
            # Print start of cross validation fold
            print('--------------------------------')
            print('Fold {}'.format(fold + 1))
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_sub_sampler = SubsetRandomSampler(train_id)
            val_sub_sampler = SubsetRandomSampler(val_id)

            # Define data loaders for training and validation data in this fold
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sub_sampler)
            val_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sub_sampler)

            # Initialize model
            model = Model()
            model = model.to(self.device)
            if fold > 0:
                model.apply(reset_weight)  # reset model's weight to avoid weight leakage
            # Initialize optimizer
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

            # Start epoch
            min_loss = math.inf
            max_accu = -1
            val_accu_list = []
            recall_list = []
            for epoch in tqdm(range(self.epochs)):
                # Load the saved model and optimizer
                if epoch > 0:
                    checkpoint = torch.load(self.tmp_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Train Phase
                train_loss, train_accu = run_train_batch(model, optimizer, train_dataloader)

                # Create checkpoint for model and optimizer before validation
                checkpoint = {
                    'model': Model(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, self.tmp_path)

                # Validation Phase
                val_loss, val_accu, recall, _, _ = run_test_batch(model, val_dataloader)
                val_accu_list.append(val_accu)
                recall_list.append(recall)

                # Save model & optimizer based on the chosen strategy
                if self.save_model:
                    model_fold_name = self.model_name[:-4] + '_fold_{}.pth'.format(fold + 1)
                    if self.strategy == 'loss':
                        min_loss = min(val_loss, min_loss)
                        if min_loss == val_loss:
                            torch.save(checkpoint, os.path.join(model_dir, model_fold_name))
                    elif self.strategy == 'accuracy':
                        max_accu = max(val_accu, max_accu)
                        if max_accu == val_accu:
                            torch.save(checkpoint, os.path.join(model_dir, model_fold_name))

                print('epoch={}'.format(epoch))
                print('train_loss={:.3f}, val_loss={:.3f}'.format(train_loss, val_loss))
                print('train_accu={:.3f}, val_accu={:.3f}, recall={:.3f}'.format(train_accu, val_accu, recall))

            # Save results
            fold_accu[fold] = val_accu_list
            fold_recall[fold] = recall_list
            self.results["recall"] = fold_recall
            self.results["val_accu"] = fold_accu

        return self.results

    @staticmethod
    def report_single_metric(metric: str, result_dict: dict):
        """
        Function to help print and plot validation results
        :param metric: metric to be plotted
        :param result_dict: dictionary containing validation results over the folds
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        result_max = []
        result_mean = []
        result_std = []
        for fold, result_list in result_dict.items():
            ax.plot(result_list, label='Fold ' + str(fold + 1))
            print('{} for Fold {}: max = {}, mean = {}, std = {:.5f}'
                  .format(metric, fold + 1, max(result_list), np.mean(result_list), np.std(result_list)))
            result_max.append(max(result_list))
            result_mean.append(np.mean(result_list))
            result_std.append(np.std(result_list))
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.set_title('{} over epoch for {}-fold cross validation'.format(metric, len(result_dict.items())))
        ax.legend()

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(0, len(result_dict.items()))
        ax1.errorbar(x, result_mean, yerr=result_std, fmt='-o')
        ax.set_xlabel('Fold')
        ax1.set_title('Mean & standard deviation of {} for {}-fold cross validation'
                      .format(metric, len(result_dict.items())))

        print('Summary of {} for {} folds:'.format(metric, len(result_dict.items())))
        print('Average max = {}, Average mean = {}, Average std = {}'
              .format(np.mean(result_max), np.mean(result_mean), np.mean(result_std)))
        print('==================================')

    def report_results(self):
        """
        Print and Plot validation results
        :return:
        """
        for metric, values in self.results.items():
            self.report_single_metric(metric, values)
