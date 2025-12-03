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

def test(model, datald, sizes, device):
    """
        Receive a trained model and
        a test dataset
        run the images in the test set
        thorugh the model and return a list
        of the most likely predictions with
        their probability and the correct result
    """
    model.eval() # set model to evaluate mode
    #test accuracy
    running_corrects = 0

    res = []
    with torch.no_grad():
        for inputs, target in datald:
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
                res.extend(model(inputs).detach().cpu().numpy().copy())
    return res

def evaluatePredictions(datald,predictions):
    """
        Receive a trained model and
        a test dataset
        run the images in the test set
        thorugh the model and return a list
        of the most likely predictions with
        their probability and the correct result
    """
    #test accuracy
    running_corrects = 0

    # create a confusion matrix
    numClasses = len(predictions[0])
    conf = np.zeros(shape=(numClasses, numClasses),dtype=np.uint8)

    for (inputs, target),outputs in zip(datald,predictions):
            #print(target)
            #print(outputs)
            pred = np.argmax(outputs)
            #print(preds)
            conf[target,pred]+=1

            #count the number of correct preds
            running_corrects += (pred == target)
    #calculation test Accuracy
    test_acc = running_corrects / len(datald)
    return test_acc,conf

def visualize_confusion_matrix(conf_matrix, class_labels=None, title="Confusion Matrix"):
    """
    Visualizes a confusion matrix using Seaborn's heatmap.
    """
    if not isinstance(conf_matrix, np.ndarray) or conf_matrix.ndim != 2 or \
       conf_matrix.shape[0] != conf_matrix.shape[1]:
        print("Invalid confusion matrix. It must be a square 2D NumPy array.")
        return

    num_classes = conf_matrix.shape[0]

    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]
    elif len(class_labels) != num_classes:
        print(f"Warning: Number of class labels ({len(class_labels)}) does not match "
              f"the number of classes in the matrix ({num_classes}). Using default labels.")
        class_labels = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(6, 4)) # Adjust figure size as needed
    sns.heatmap(
        conf_matrix,
        annot=True,      # Annotate cells with the numeric values
        fmt="d",         # Format annotations as integers
        cmap="Blues",    # Color map (e.g., "Blues", "viridis", "YlGnBu")
        cbar=True,       # Show color bar
        xticklabels=class_labels, # Labels for predicted classes
        yticklabels=class_labels # Labels for true classes
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

def interactive_evaluation(datald, predictions):
    """
    Interactively evaluates a model's predictions, compares with user input,
    displays results, accuracy, and confusion matrices.
    """
    # Determine the number of classes from the first prediction's length
    num_classes = len(predictions[0])

    # Initialize metrics for user and model
    running_corrects_user = 0
    conf_user = np.zeros(shape=(num_classes, num_classes), dtype=np.uint8)

    running_corrects_model = 0
    conf_model = np.zeros(shape=(num_classes, num_classes), dtype=np.uint8)

    total_evaluated_images = 0

    print("\n--- Starting Interactive Evaluation ---")
    print("Enter the category number for the image, or 'q' to quit at any time.")

    # Enable interactive mode for matplotlib to prevent plots from blocking input
    plt.ion()

    # Iterate through the dataset and corresponding predictions
    for i, ((image_data, true_target), model_output) in enumerate(zip(datald, predictions)):
        total_evaluated_images += 1

        # Get the model's predicted class (index of the highest score/probability)
        model_pred = np.argmax(model_output)

        display_image = image_data # Start with the image data as returned by __getitem__

        # Check if the image is in (C, H, W) format (channels-first)
        if display_image.ndim == 3 and (display_image.shape[0] == 3 or display_image.shape[0] == 1):
            # Transpose to (H, W, C) for matplotlib display
            display_image = np.transpose(display_image, (1, 2, 0))
            # If it's a single channel image (1, H, W), remove the channel dimension to make it (H, W) for imshow
            if display_image.shape[2] == 1:
                display_image = display_image.squeeze(axis=2)

        # Convert BGR to RGB if it's a 3-channel image.
        # Assuming the original cv2.imread in Classification_Dataset returns BGR,
        # and no BGR2RGB conversion was done before the transpose.
        if display_image.ndim == 3 and display_image.shape[2] == 3:
            # If image_data was not normalized to [0,1] in __getitem__, then it's 0-255.
            # Convert to uint8 for cv2.cvtColor, then back to float32 and normalize for display.
            if display_image.max() > 1.0: # Check if it's in 0-255 range
                display_image = cv2.cvtColor(display_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                display_image = display_image.astype(np.float32) / 255.0 # Normalize for matplotlib
            else: # Already normalized [0,1], assume it's still BGR and convert
                # Multiply by 255, convert to uint8, convert color, then divide by 255 again
                display_image = cv2.cvtColor((display_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                display_image = display_image.astype(np.float32) / 255.0


        # Create a new figure for each image
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(display_image)
        plt.title(f"Image {i+1}/{len(datald)}")
        plt.axis('off') # Hide axes ticks
        plt.show(block=False) # Show the plot without blocking the execution
        plt.pause(0.1) # Give a short pause to allow the plot to render

        user_input_valid = False
        user_pred = -1
        while not user_input_valid:
            try:
                user_input = input(f"Enter your category (0-{num_classes-1}) or 'q' to quit: ").strip().lower()
                if user_input == 'q':
                    user_input_valid = True # Exit inner loop, then break outer loop
                else:
                    user_pred = int(user_input)
                    if 0 <= user_pred < num_classes:
                        user_input_valid = True
                    else:
                        print(f"Invalid category. Please enter a number between 0 and {num_classes-1}.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")

        plt.close(fig) # Close the current image figure to free up resources

        if user_input == 'q':
            print("\n--- User quit interactive evaluation. ---")
            total_evaluated_images -= 1 # Do not count the image if the user quit on it
            break

        # Update user's confusion matrix and correct count
        conf_user[true_target, user_pred] += 1
        running_corrects_user += (user_pred == true_target)

        # Update model's confusion matrix and correct count
        conf_model[true_target, model_pred] += 1
        running_corrects_model += (model_pred == true_target)

        print(f"--- Results for Image {i+1} ---")
        print(f"True Label: {true_target}")
        print(f"Your Prediction: {user_pred} {'(Correct)' if user_pred == true_target else '(Incorrect)'}")
        print(f"Model Prediction: {model_pred} {'(Correct)' if model_pred == true_target else '(Incorrect)'}")
        print("-" * 30)

    plt.ioff() # Turn off interactive mode after the loop

    if total_evaluated_images == 0:
        print("No images were evaluated.")
        return

    # Calculate final accuracies
    user_accuracy = running_corrects_user / total_evaluated_images
    model_accuracy = running_corrects_model / total_evaluated_images

    print("\n--- Evaluation Summary ---")
    print(f"Total images evaluated: {total_evaluated_images}")
    print(f"Your Accuracy: {user_accuracy:.2f}")
    print(f"Model Accuracy: {model_accuracy:.2f}")

    # Visualize confusion matrices using the previously defined function
    visualize_confusion_matrix(conf_user, title="Your Confusion Matrix")
    visualize_confusion_matrix(conf_model, title="Model's Confusion Matrix")
