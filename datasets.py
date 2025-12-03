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

class Classification_Dataset(Dataset):

    def percentage_to_class(self, percentage):
        cat = int(percentage * self.n_classes)
        return min(cat, self.n_classes - 1)

    def __init__(self, imageFolder=None, dataFile=None, n_classes = 10):
        self.imageList = []
        self.labelList = []
        self.imageNames = []
        self.n_classes = n_classes
        self.dataDict = {}

        self.imageFolder = imageFolder
        self.dataFile = dataFile

        if imageFolder is None or dataFile is None:
            return

        with open(dataFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.dataDict[parts[0]] = float(parts[1])

        for k, v in self.dataDict.items():
          self.imageNames.append(k)
          cat = self.percentage_to_class(v / 100)
          self.labelList.append(cat)


    def __getitem__(self, index):
        image_path = os.path.join(self.imageFolder, self.imageNames[index])
        currentImage = cv2.imread(image_path)
        if currentImage is None:
            raise Exception("Error reading image: " + image_path)
        image = np.moveaxis(currentImage, -1, 0).astype(np.float32)
        target = self.labelList[index]

        return image, target

    def __len__(self):
        return len(self.imageNames)

    def numClasses(self): return self.n_classes

    #Create two dataset (training and validation) from an existing one, do a random split
    def breakTrainValid(self, proportion):
        train = Classification_Dataset(None)
        valid = Classification_Dataset(None)
        train.n_classes = self.n_classes
        valid.n_classes = self.n_classes

        train.imageFolder = self.imageFolder
        valid.imageFolder = self.imageFolder
        train.dataDict = self.dataDict.copy()
        valid.dataDict = self.dataDict.copy()

        toDivide = random.sample(list(zip(self.imageNames, self.labelList)), len(self.imageNames))

        for i in range(int(len(self) * proportion)):
          valid.imageNames.append(toDivide[i][0])
          valid.labelList.append(toDivide[i][1])

        for i in range(int(len(self) * proportion), len(self)):
          train.imageNames.append(toDivide[i][0])
          train.labelList.append(toDivide[i][1])

        return train, valid
def label_to_text(label, step=10):
    start = label * step
    end = start + step - 0.01
    return f"{start:.2f}% – {end:.2f}%"

def showdataSetWithTargets(dataset, num_images_to_show = 10):
    """
    Displays a random selection of images from a Classification_Dataset
    """
    n_cols = min(num_images_to_show, 5) # Max 5 columns for better layout
    n_rows = (num_images_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() # Flatten the axes array for easy iteration

    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_images_to_show, len(dataset)))

    for i, idx in enumerate(indices):
      img, target = dataset[idx]     # C×H×W, float32, 0–1

      # Transpose and change to RGB
      img = np.moveaxis(img, 0, -1).astype(np.float32)
      img = img[..., ::-1]

      axes[i].imshow(img)
      axes[i].set_title(f"Target {int(target)}\n{label_to_text(int(target))}")
      axes[i].axis("off")
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to generate a chessboard image and compute red cell percentage
def generate_chessboard_percentage(grid_size, cell_size, main_color = (0, 0, 255), bck_color = (255, 0, 0) ):
    rows, cols = grid_size

    # Generate boolean mask with per-cell random selection
    mask = np.random.rand(rows, cols) < random.uniform(0, 1)
    count = mask.sum()

    # Create a color grid based on the mask
    color_grid = np.where(mask[..., None], main_color, bck_color).astype(np.uint8)

    # Scale up each cell into a tile using np.kron
    image = np.kron(color_grid, np.ones((cell_size, cell_size, 1), dtype=np.uint8))

    return (count / (rows * cols)) * 100, image

def makeSet(num_images = 100, grid_rows = 50, grid_cols = 50, cell_dimension = 10, outText = os.path.join('datasetD1', 'trainData.txt'),
          outPath = os.path.join('datasetD1', 'training' ), mc = (0, 0, 255), bc = (255, 0, 0)):
    """
        use the generate_chessboard_percentage
        to create datasets
    """
    image_files = []
    percentage_data = []

    # make folder if it did not exist
    Path(outPath).mkdir(parents=True, exist_ok=True)

    # Generate images
    for i in range(num_images):
        percentage, chessboard_image = generate_chessboard_percentage((grid_rows, grid_cols), cell_dimension,
                                                                      main_color = mc, bck_color = bc)
        output_filename = f"im{i}.png"
        full_path = os.path.join(outPath, output_filename)
        cv2.imwrite(full_path, chessboard_image)
        image_files.append(output_filename)
        percentage_data.append({'filename': output_filename, 'percentage': percentage})

    # Save data as text file
    with open(outText, 'w') as f:
      for entry in percentage_data:
          f.write(f"{entry['filename']} {entry['percentage']:.2f}\n")
