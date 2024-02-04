import torchvision.transforms.functional as TF
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler, BatchSampler
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight
from skimage import filters, segmentation, measure

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)])
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.CenterCrop((512, 512))])
trans1 = transforms.Compose([
    # transforms.RandomCrop(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trans2 = transforms.Compose([
    # transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor()])

trans3 = transforms.Compose([
    # transforms.RandomCrop(256),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor()])
class_name = ["NRG", "RG"]
train_image_path = os.path.join(
    "./data/", "ISBI_2024/resize_512_images/")
train_gt_path = os.path.join(
    "./data/", "ISBI_2024", "JustRAIGS_Train_labels.csv")

train_gt_pdf = pd.read_csv(train_gt_path, delimiter=';')
# train_gt_pdf = train_gt_pdf[:100]
train_image_name = train_gt_pdf["Eye ID"]
train_label_list = train_gt_pdf.iloc[:, 1:].apply(
    lambda row: {col.lower(): row[col] for col in train_gt_pdf.columns[1:]}, axis=1).tolist()

input_data = train_gt_pdf['Eye ID']
labels = train_gt_pdf['Final Label']
# Map class labels to numerical values
class_to_numeric = {class_label: idx for idx,
                    class_label in enumerate(class_name)}

# Transform labels into numerical format (0 or 1)
labels_numeric = [class_to_numeric[label] for label in labels]
labels_numeric = np.array(labels_numeric)
# Choose fold to train on
kf = KFold(n_splits=5,
           shuffle=True, random_state=111)

all_splits = [k for k in kf.split(input_data, labels_numeric)]

train_indexes, val_indexes = all_splits[0]

# Count the number of samples in class 1 in the training set
train_input_data = input_data[train_indexes]
train_label_data = labels_numeric[train_indexes]

val_input_data = input_data[val_indexes]
val_label_data = labels_numeric[val_indexes]

train_class_counts = np.bincount(train_label_data)
val_class_counts = np.bincount(val_label_data)

print("train/class_zeros_count: ", train_class_counts[0])
print("train/class_ones_count: ", train_class_counts[1])
print("val/class_zeros_count: ",
      val_class_counts[0])
print("val/class_ones_count: ", val_class_counts[1])


class IsbiDataSet(Dataset):
    def __init__(self, data, label, class_name, data_length, data_dir, train_image_path, is_transform, transform, is_training, image_size):
        super().__init__()
        self.data = data
        self.label = label
        self.class_name = class_name
        self.train_image_path = train_image_path
        self.data_length = data_length
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.transform = transform
        self.is_training = is_training
        self.image_size = image_size

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Load image using df['image'][index], assuming 'image' is the column containing image paths
        image = None
        try:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.train_image_path, self.data[index] + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed

        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.train_image_path, self.data[index] + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed

            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.train_image_path, self.data[index] + ".jpeg")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    image = Image.open(image_path).convert(
                        'RGB')  # Adjust as needed

                except FileNotFoundError:
                    # Handle the case where both .jpg and .png files are not found
                    print(f"Error: File not found for index {index}")
                    # You might want to return a placeholder image or raise an exception as needed

        # Apply transformations if specified
        if image is not None:
            if self.is_transform:
                image = self.transform(image)

            else:
                transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                image = self.transform(image)
            # else:
            #     transforms = transforms.Compose([
            #         transforms.Resize((image_size, image_size)),
            #         transforms.ToTensor(),
            #         transforms.Normalize((0.1307,), (0.3081,))
            #     ])
            #     image = transform(image)
            # Extract class labels, assuming 'MEL', 'NV', etc., are columns in your CSV file
            label = self.label[index]

            # Create a one-hot encoded tensor
            if len(class_name) > 2:
                one_hot_encoded = torch.zeros(
                    len(class_name), dtype=torch.float32)
                one_hot_encoded[label] = torch.ones(1, dtype=torch.float32)
            else:
                one_hot_encoded = torch.tensor(label, dtype=torch.float32)

            return image, one_hot_encoded
        else:
            return None


def crop_to_shape(arr, shape, cval=0):
    """Crops a numpy array into the specified shape. If the array was larger, return centered crop. If it was smaller,
    return a larger array with the original data in the center"""
    if arr.ndim != len(shape):
        raise Exception("Array and crop shape dimensions do not match")

    arr_shape = np.array(arr.shape)
    shape = np.array(shape)
    max_shape = np.stack([arr_shape, shape]).max(axis=0)
    output_arr = np.ones(max_shape, dtype=arr.dtype) * cval

    arr_min = ((max_shape - arr_shape) / 2).astype(np.int32)
    arr_max = arr_min + arr_shape
    slicer_obj = tuple(slice(idx_min, idx_max, 1)
                       for idx_min, idx_max in zip(arr_min, arr_max))
    output_arr[slicer_obj] = arr

    crop_min = ((max_shape - shape) / 2).astype(np.int32)
    crop_max = crop_min + shape
    slicer_obj = tuple(slice(idx_min, idx_max, 1)
                       for idx_min, idx_max in zip(crop_min, crop_max))
    # Return a copy of the view, so the rest of memory can be GC
    return output_arr[slicer_obj].copy()


def crop_retina(image):
    """Return a square crop of the image centered on the retina.
    This function does the following assumtions:
    - image is an np.array with dimensions [height, weight, channels] or [height, weight]
    - the background of the retinography will have a stark contrast with the rest of the image
    """
    # Check dimensionality of the array is valid
    if image.ndim > 3:
        raise Exception("image has too many dimensions. Max 3")
    elif image.ndim < 2:
        raise Exception("image has too few dimensions. Min 2")

    # Rescale image to ensure there will be a black border around (even if the original was already cropped)
    image = crop_to_shape(
        image,
        np.array(image.shape) + np.array([20, 20, 0])[:image.ndim],
        cval=0
    )

    # If image is an RGB array, convert to grayscale
    if image.ndim == 3:
        bw_image = np.mean(image, axis=-1)
    else:
        bw_image = image

    # Find and apply threshold, to create a binary mask
    thresh = filters.threshold_triangle(bw_image)
    binary = bw_image > thresh

    # Label image regions and select the largest one (the retina)
    label_image = measure.label(binary)
    eye_region = sorted(measure.regionprops(
        label_image), key=lambda p: -p.area)[0]

    # Crop around the retina
    y_start, x_start, y_end, x_end = eye_region.bbox
    y_diff = y_end - y_start
    x_diff = x_end - x_start
    if x_diff > y_diff:
        if (y_start + x_diff) <= binary.shape[0]:
            y_end_x_diff = (y_start + x_diff)
            cropped_image = image[y_start:y_end_x_diff, x_start:x_end]
        else:
            y_start_x_diff = (y_end - x_diff) if (y_end - x_diff) > 0 else 0
            cropped_image = image[y_start_x_diff:y_end, x_start:x_end]
    else:
        if (x_start + y_diff) <= binary.shape[1]:
            x_end_y_diff = (x_start + y_diff)
            cropped_image = image[y_start:y_end, x_start:x_end_y_diff]
        else:
            x_start_y_diff = (x_end - y_diff) if (x_end - y_diff) > 0 else 0
            cropped_image = image[y_start:y_end, x_start_y_diff:x_end]

    # Ensure aspect ratio will be square
    max_axis = max(cropped_image.shape)
    if cropped_image.ndim == 3:
        square_crop = (max_axis, max_axis, cropped_image.shape[-1])
    else:
        square_crop = (max_axis, max_axis)
    square_image = crop_to_shape(cropped_image, square_crop)
    return square_image


data_train = IsbiDataSet(
    train_input_data.tolist(), train_label_data.tolist(), class_name, len(train_input_data), "./data/", train_image_path, True, trans, is_training=True, image_size=512)

data_val = IsbiDataSet(
    val_input_data.tolist(), val_label_data.tolist(), class_name, len(val_input_data), "./data/", train_image_path, True, trans, is_training=False, image_size=512)


def visualize_images(dataset, label_value=1, num_images=5):

    for index, data in enumerate(dataset):
        image = data[0]
        label = data[1]
        if label != label_value:
            continue
        else:
            # Convert tensor to PIL Image for visualization
            image = TF.to_pil_image(image)
            image = crop_retina(np.array(image))
            plt.imshow(image)
            plt.show()


# Assuming 'data_train' is an instance of IsbiDataSet
visualize_images(data_train, label_value=1, num_images=2000)
