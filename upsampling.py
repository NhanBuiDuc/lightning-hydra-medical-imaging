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
        hue_saturation_image = original_image.copy()
        gray_scale_image = original_image.copy()
        posterize_image = original_image.copy()
        solarize_image = original_image.copy()
        sharp_image = original_image.copy()
        contrast_image = original_image.copy()
        equalize_image = original_image.copy()
        invert_image = original_image.copy()
        affine_image = original_image.copy()
        rotation_image = original_image.copy()
        vertical_flip_image = original_image.copy()
        horizontal_flip_image = original_image.copy()

        # Geometric Augmentation: Random rotation, vertical flip, and horizontal flip

        rotation = transforms.RandomRotation((180, 180))

        horizontal_flip = transforms.RandomHorizontalFlip(p=1)

        vertical_flip = transforms.RandomVerticalFlip(p=1)

        affine = transforms.RandomAffine(degrees=(-180, 180))
        rotation_image = rotation(rotation_image)
        vertical_flip_image = horizontal_flip(rotation_image)
        horizontal_flip_image = vertical_flip(rotation_image)
        affine_image = affine(affine_image)
        rotation_image.save(os.path.join(
            geo_directory, f"{image_name}_rotated_geo.jpg"))
        horizontal_flip_image.save(os.path.join(
            geo_directory, f"{image_name}_horizontal_flip_geo.jpg"))
        vertical_flip_image.save(os.path.join(
            geo_directory, f"{image_name}_vertical_flip_geo.jpg"))
        affine_image.save(os.path.join(
            geo_directory, f"{image_name}_affine_geo.jpg"))

        # Color Augmentation: Hue = 0.5, Saturation = 0.5
        hue_saturation_transform = transforms.Compose([
            transforms.ColorJitter(hue=0.5, saturation=0.5)
        ])
        gray_scale_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3)
        ])
        invert_transform = transforms.Compose([
            transforms.RandomInvert(p=1)
        ])
        posterize_transform = transforms.Compose([
            transforms.RandomPosterize(bits=3, p=1)
        ])
        solarize_transform = transforms.Compose([
            transforms.RandomSolarize(threshold=0.6, p=1)
        ])
        sharp_transform = transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1)
        ])
        contrast_transform = transforms.Compose([
            transforms.RandomAutocontrast(p=1)
        ])
        equalize_transform = transforms.Compose([
            transforms.RandomEqualize(p=1)
        ])

        hue_saturation_image = hue_saturation_transform(hue_saturation_image)
        gray_scale_image = gray_scale_transform(gray_scale_image)
        invert_image = invert_transform(invert_image)
        posterize_image = posterize_transform(posterize_image)
        solarize_image = solarize_transform(solarize_image)
        sharp_image = sharp_transform(sharp_image)
        contrast_image = contrast_transform(contrast_image)
        equalize_image = equalize_transform(equalize_image)

        hue_saturation_image.save(os.path.join(
            color_directory, f"{image_name}_hue_saturation_color.jpg"))
        gray_scale_image.save(os.path.join(
            color_directory, f"{image_name}_gray_color.jpg"))
        invert_image.save(os.path.join(
            color_directory, f"{image_name}_invert_color.jpg"))
        posterize_image.save(os.path.join(
            color_directory, f"{image_name}_posterize_color.jpg"))
        solarize_image.save(os.path.join(
            color_directory, f"{image_name}_solarize_color.jpg"))
        sharp_image.save(os.path.join(
            color_directory, f"{image_name}_sharp_color.jpg"))
        contrast_image.save(os.path.join(
            color_directory, f"{image_name}_contrast_color.jpg"))
        equalize_image.save(os.path.join(
            color_directory, f"{image_name}_equalize_color.jpg"))
