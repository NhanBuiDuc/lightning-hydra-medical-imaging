from PIL import Image
import os
import pandas as pd
import numpy as np
import random
import torchvision.transforms as transforms

class_name = ["NRG", "RG"]
train_image_path = os.path.join(
    "./data/", "ISBI_2024/resize_512_images/")
train_gt_path = os.path.join(
    "./data/", "ISBI_2024", "JustRAIGS_Train_labels.csv")

train_gt_pdf = pd.read_csv(train_gt_path, delimiter=';')

# Assuming 'Eye ID' is the column with image names, and 'Final Label' is the column with labels
input_data = train_gt_pdf['Eye ID']
labels = train_gt_pdf['Final Label']

# Map class labels to numerical values
class_to_numeric = {class_label: idx for idx,
                    class_label in enumerate(class_name)}

# Transform labels into numerical format (0 or 1)
labels_numeric = [class_to_numeric[label] for label in labels]
labels_numeric = np.array(labels_numeric)

# Output directories for augmented images
geo_directory = "geo_aug_images/"
color_directory = "color_aug_images/"

# Create the output directories if they don't exist
os.makedirs(geo_directory, exist_ok=True)
os.makedirs(color_directory, exist_ok=True)

# Loop through images with label "RG" (class label 1)
for idx, (image_name, label) in enumerate(zip(input_data, labels_numeric)):
    if label == 1:  # Check if the label is "RG"
        # Assuming images have .jpg extension
        input_path = os.path.join(train_image_path, f"{image_name}.jpg")
        try:
            # Open the original image
            original_image = Image.open(input_path)
        except FileNotFoundError:
            try:
                # Try opening with .png extension
                input_path = os.path.join(
                    train_image_path, f"{image_name}.png")
                original_image = Image.open(input_path)
            except FileNotFoundError:
                try:
                    # Try opening with .jpeg extension
                    input_path = os.path.join(
                        train_image_path, f"{image_name}.jpeg")
                    original_image = Image.open(input_path)
                except FileNotFoundError:
                    # Handle the case when the image file is not found
                    print(f"Image file for {image_name} not found.")
                    continue  # Skip to the next iteration or add your desired logic
        # Open the original image
        original_image = Image.open(input_path)

        rotation_image = original_image.copy()
        vertical_flip_image = original_image.copy()
        horizontal_flip_image = original_image.copy()
        # Geometric Augmentation: Random rotation, vertical flip, and horizontal flip

        rotation = transforms.RandomRotation((180, 180))

        horizontal_flip = transforms.RandomHorizontalFlip(p=1)

        vertical_flip = transforms.RandomVerticalFlip(p=1)

        rotation_image = rotation(rotation_image)
        vertical_flip_image = horizontal_flip(rotation_image)
        horizontal_flip_image = vertical_flip(rotation_image)

        rotation_image.save(os.path.join(
            geo_directory, f"{image_name}_rotated_image.jpg"))
        horizontal_flip_image.save(os.path.join(
            geo_directory, f"{image_name}_horizontal_flip_image.jpg"))
        vertical_flip_image.save(os.path.join(
            geo_directory, f"{image_name}_vertical_flip_image.jpg"))
        # Color Augmentation: Hue = 0.5, Saturation = 0.5
        color_transforms = transforms.Compose([
            transforms.ColorJitter(hue=0.5, saturation=0.5)
        ])

        color_augmented_image = color_transforms(original_image)

        # Save the color augmented image
        color_output_path = os.path.join(
            color_directory, f"{image_name}_color_aug.jpg")
        color_augmented_image.save(color_output_path)

# Note: Adjust the augmentation parameters based on your specific requirements.
