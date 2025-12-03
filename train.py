# general imports
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
#import re
from pathlib import Path
import copy # for the deep copy of deep learning models
import time
import random
import seaborn as sns

# Pytorch imports
import torch
import torch.nn as nn # basic pytorch module
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim # Pytorch optimizers
from torchvision import transforms, models # Pytorch pre-defined models


def duplicate_rename(file_path):
    if os.path.exists(file_path):
        name, ext = os.path.splitext(file_path)
        i = 1
        while True:
            new_name = '{} ({}){}'.format(name, i, ext)
            if not os.path.exists(new_name):
                return new_name
            i += 1
    else:
        return file_path


def train_valid(model, criterion, optimizer, dataloader, sizes, device):

    running_loss = 0.0
    running_corrects = 0
    data_count=0

    for inputs, target in dataloader:
        #print("starting interation data "+str(target))
        #print("starting interation target "+str(target.data))

        inputs, target = inputs.to(device), target.to(device)
        data_count+=len(inputs)

        # forward pass
        outputs = model(inputs)

        #print("model outputs after forward "+str(outputs))
        _, preds = torch.max(outputs, 1)

        #print("model preds "+str(preds))

        # loss function
        loss = criterion(outputs, target)

        if model.training:

            optimizer.zero_grad() # zero the parameter gradients

            # back-propagation
            loss.backward()

            # optimization calculations
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        #print(preds)
        #print(target)
        running_corrects += torch.sum(preds == target.data)


    #epoch loss and accuracy
    epoch_loss = running_loss / sizes
    epoch_acc = running_corrects / sizes
    epoch_acc=epoch_acc.to('cpu').detach().numpy().copy()
    return epoch_loss, epoch_acc

def optimize_model(model, criterion, optimizer, dataloaderTrain, dataloaderTest, sizeTrain, sizeTest,
                   device, num_epochs=25, save_model = True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[],}

    print('training start !!!!!')
    print("-"*64)

    for epoch in range(num_epochs):
        string = f'| epoch {epoch + 1}/{num_epochs}:'+" "*50+"|"
        print(string)

        with torch.set_grad_enabled(True):
            model.train() # set model to training mode
            train_loss, train_acc = train_valid(model, criterion, optimizer,  dataloaderTrain, sizeTrain, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        with torch.no_grad():
            model.eval() # set model to evaluation (validation) mode
            val_loss, val_acc = train_valid(model, criterion, None, dataloaderTest, sizeTest, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("| train loss:  {:.4f} | train acc:  {:.4f} | val acc:  \033[31m{:.4f}\033[0m  |".format(train_loss, train_acc, val_acc))
        else:
            print("| train loss:  {:.4f} | train acc:  {:.4f} | val acc:  {:.4f}  |".format(train_loss, train_acc,val_acc))
        print('-'*64)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:6f}'.format(best_acc))

    # save best model weights
    pth = f'best_wts_{model.__class__.__name__}50_epo'+str(num_epochs)+'.pth'
    #new_pth = duplicate_rename(pth)
    new_pth = pth # overwrite results

    if save_model:
        torch.save(best_model_wts, new_pth)
        print("model parameters was saved...")


    return best_acc, history

def plot_loss_acc(history, model, num_epochs):
    """
    Function to plot the train
    and validation losses
    """
    epochs = np.arange(1, num_epochs + 1)
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    # transition of loss
    ax1.set_title("Loss")
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    # transition of accuracy
    ax2.set_title("Acc")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.show()
    #plt.savefig(f'result_{model.__class__.__name__}.jpg')
