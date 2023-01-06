from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import recall_score


def train_step(model, optimizer, sample,
               device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    """
    Function to run one step of training a model
    :param model: model to be trained
    :param optimizer: optimizer used for training
    :param sample: input sample
    :param device: default to GPU
    :return:
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    inp = sample['input'].float().to(device)
    target = sample['label'].long().to(device)

    pred = model(inp)
    pred_loss = criterion(pred, target)

    top_val, top_idx = torch.topk(pred, 1)
    num_correct = torch.sum(top_idx == target.view(-1, 1))

    pred_loss.backward()
    optimizer.step()

    return pred_loss.item(), num_correct.item()


def run_train_batch(model, optimizer, dataloader):
    """
    Function to run training code for one batch
    :param model: model to be trained
    :param optimizer: optimizer used for training
    :param dataloader: input data generator
    :return:
    """
    # Initialize
    loss = 0.0
    total = 0.0
    correct = 0.0
    # Iterate over dataloader
    with tqdm(total=len(dataloader)) as pbar:
        for idx, sample in enumerate(dataloader):
            curr_loss, num_correct = train_step(model, optimizer, sample)
            loss += curr_loss / len(dataloader)
            total += sample['label'].size(0)
            correct += num_correct
            pbar.update(1)
        accu = correct / total

    return loss, accu


def run_test_batch(model, dataloader):
    """
    Function to run evaluation for one batch
    :param model: model to be evaluated
    :param dataloader: input data generator
    :return:
    """
    # Initialize
    loss = 0.0
    total = 0.0
    correct = 0.0
    y_pred = []
    y_true = []
    # Iterate over dataloader
    with tqdm(total=len(dataloader)) as pbar:
        for idx, sample in enumerate(dataloader):
            curr_loss, num_correct, pred = test_step(model, sample)
            loss += curr_loss / len(dataloader)
            total += sample['label'].size(0)
            correct += num_correct
            _, pred_label = torch.max(pred, 1)
            y_pred.append(pred_label.cpu())
            y_true.append(sample['label'].cpu())
            pbar.update(1)
        accu = correct / total
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        recall = recall_score(y_true, y_pred, average='macro')

    return loss, accu, recall, y_pred, y_true


def test_step(model, sample, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    """
    Function to run one step of testing a model. The model's layers are frozen during testing
    :param model: model to be tested
    :param sample: input sample
    :param device: default to GPU
    :return:
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        inp = sample['input'].float().to(device)
        target = sample['label'].long().to(device)

        pred = model(inp)
        pred_loss = criterion(pred, target)

        top_val, top_idx = torch.topk(pred, 1)

        num_correct = torch.sum(top_idx == target.view(-1, 1))

    return pred_loss.item(), num_correct.item(), pred


def calc_recall(pred: torch.Tensor, true: torch.Tensor):
    """
    Calculate recall
    :param pred: model's prediction
    :param true: true labels
    :return:
    """
    assert true.ndim == 1

    if pred.ndim > 1:
        pred = pred.argmax(dim=1)

    tp = (pred == true).sum().to(torch.float32)
    fn = (true == (1 - pred)).sum().to(torch.float32)
    epsilon = 1e-7  # To avoid division by zero

    recall = tp / (tp + fn + epsilon)

    return recall


def reset_weight(model):
    """
    Reset model's weights to avoid weight leakage
    :param model: model whose weights are to be reset
    :return:
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def load_checkpoint(model_path, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    """
    Load a PyTorch model
    :param model_path: path to model.pth
    :param device: default to GPU
    :return:
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model


def print_run_config(epochs, train_strategy, save_model, model_dir, model_name, device):
    """
    Print hyperparameter configurations
    :param epochs: number of epochs
    :param train_strategy: two possible strategies, 'loss' or 'accu':
                           'loss' save the best model with minimum loss,
                           'accu' save the best model with maximum accuracy
    :param save_model: bool to save model
    :param model_dir: path to save the model
    :param model_name: name of the model to be saved
    :param device: device to be used (GPU or CPU)
    :return:
    """
    print("Device : {}".format(device))
    print("Training for {} epochs".format(epochs))
    if save_model:
        if train_strategy == 'loss':
            print("Strategy: save best model in term of minimum loss")
        elif train_strategy == 'accuracy':
            print("Strategy: save best model in term of maximum accuracy")
        else:
            print("Invalid training strategy")
        print("The best model will be saved under the name {} in folder {}".format(model_name, model_dir))
