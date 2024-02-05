from typing import Any, Dict, Optional, Tuple

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


class IsbiDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        image_size: int = 512,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 111,
        kfold_index: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        is_transform=True,
        balance_data=True,
        binary_unbalance_train_ratio=100
    ) -> None:
        """Initialize a `IsicDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        torch.backends.bottleneck = True
        # data transformations
        # self.transforms = transforms.Compose([
        #     transforms.RandomApply(
        #         [transforms.RandomCrop((image_size, image_size)), transforms.CenterCrop((image_size, image_size)), transforms.Pad(10)]),

        #     transforms.RandomApply([transforms.RandomPerspective(), transforms.RandomRotation(degrees=(
        #         0, 180)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomAffine(30), transforms.ElasticTransform()]),

        #     transforms.RandomApply([transforms.RandomGrayscale(), transforms.ColorJitter(
        #         brightness=.5, hue=.3), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), transforms.RandomInvert(), transforms.RandomPosterize(bits=5),
        #         transforms.RandomSolarize(threshold=192.0), transforms.RandomAdjustSharpness(sharpness_factor=10), transforms.RandomAutocontrast(), transforms.RandomEqualize()]),

        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),
        #     # Convert to PyTorch tensor
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        train_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        val_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
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

        self.train_transforms = train_trans
        self.val_transforms = val_trans
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_name = class_name
        self.num_class = num_class
        self.batch_size = batch_size
        self.kfold_seed = kfold_seed
        self.kfold_index = kfold_index
        self.training_split = train_val_test_split[0]
        self.validation_split = train_val_test_split[1]
        self.is_transform = is_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.balance_data = balance_data
        self.train_image_path = os.path.join(
            self.data_dir, "ISBI_2024/resize_512_images/")
        self.train_gt_path = os.path.join(
            self.data_dir, "ISBI_2024", "JustRAIGS_Train_labels.csv")
        self.geo_aug_images = os.path.join(
            self.data_dir, "ISBI_2024", "geo_aug_images")
        self.color_aug_images = os.path.join(
            self.data_dir, "ISBI_2024", "color_aug_images")

        self.train_gt_path = self.train_gt_path.replace("\\", "/")

        self.binary_unbalance_train_ratio = binary_unbalance_train_ratio

    @property
    def num_classes(self) -> int:

        return self.num_class

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Load the CSV file into a pandas DataFrame
        self.train_gt_pdf = pd.read_csv(self.train_gt_path, delimiter=';')
        # self.train_gt_pdf = self.train_gt_pdf[:100]
        self.train_image_name = self.train_gt_pdf["Eye ID"]
        self.train_label_list = self.train_gt_pdf.iloc[:, 1:].apply(
            lambda row: {col.lower(): row[col] for col in self.train_gt_pdf.columns[1:]}, axis=1).tolist()

        self.class_distribution = self.calculate_class_distribution()
        # # Calculate the number of samples for each split
        self.total_samples = int(sum(self.class_distribution.values()))

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if not self.data_train and not self.data_val:
            if not self.balance_data:
                input_data = self.train_gt_pdf['Eye ID']
                labels = self.train_gt_pdf['Final Label']
                # Map class labels to numerical values
                class_to_numeric = {class_label: idx for idx,
                                    class_label in enumerate(self.class_name)}

                # Transform labels into numerical format (0 or 1)
                labels_numeric = [class_to_numeric[label] for label in labels]
                labels_numeric = np.array(labels_numeric)
                # Choose fold to train on
                kf = KFold(n_splits=5,
                           shuffle=True, random_state=self.kfold_seed)

                all_splits = [k for k in kf.split(input_data, labels_numeric)]

                train_indexes, val_indexes = all_splits[self.kfold_index]

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

                # # Calculate class weights
                # train_class_weights = 1. / \
                #     torch.tensor(train_class_counts, dtype=torch.float)
                # # Map class labels to indices
                # train_class_to_index = {
                #     self.class_name[i]: i for i in range(len(self.class_name))}

                # train_label_indices = [train_class_to_index[label]
                #                        for label in train_label_data]

                # # Assign weights to each sample in the validation set
                # train_weights = train_class_weights[train_label_indices]

                # Assuming you have WeightedRandomSampler, you can use it like this:
                self.weighted_sampler_train = WeightedRandomSampler(
                    weights=[(0.8*self.batch_size) // 100,
                             (0.2*self.batch_size) // 100],
                    num_samples=len(train_label_data),
                    replacement=False
                )

                # Calculate class weights
                # val_class_weights = 1. / \
                #     torch.tensor(val_class_counts, dtype=torch.float)

                # Assuming you have WeightedRandomSampler, you can use it like this:
                self.weighted_sampler_val = WeightedRandomSampler(
                    weights=[(0.8*self.batch_size) // 100,
                             (0.2*self.batch_size) // 100],
                    num_samples=len(val_label_data),
                    replacement=False
                )

                self.data_train = IsbiDataSet(
                    combined_train_data, combined_label_data, self.class_name, len(combined_train_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=True, image_size=self.image_size)

                self.data_val = IsbiDataSet(
                    val_input_data.tolist(), val_label_data.tolist(), self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=False, image_size=self.image_size)
            else:
                if self.balance_data:
                    input_data = self.train_gt_pdf['Eye ID']
                    labels = self.train_gt_pdf['Final Label']
                    # Map class labels to numerical values
                    class_to_numeric = {class_label: idx for idx,
                                        class_label in enumerate(self.class_name)}

                    # Transform labels into numerical format (0 or 1)
                    labels_numeric = [class_to_numeric[label]
                                      for label in labels]
                    labels_numeric = np.array(labels_numeric)
                    # Choose fold to train on
                    kf = KFold(n_splits=5,
                               shuffle=True, random_state=self.kfold_seed)

                    all_splits = [k for k in kf.split(
                        input_data, labels_numeric)]

                    train_indexes, val_indexes = all_splits[self.kfold_index]

                    # Count the number of samples in class 1 in the training set
                    train_input_data = input_data[train_indexes]
                    train_label_data = labels_numeric[train_indexes]

                    val_input_data = input_data[val_indexes]
                    val_label_data = labels_numeric[val_indexes]

                    train_class_counts = np.bincount(train_label_data)
                    val_class_counts = np.bincount(val_label_data)

                    print("original_train/class_zeros_count: ",
                          train_class_counts[0])
                    print("original_train/class_ones_count: ",
                          train_class_counts[1])
                    print("original_val/class_zeros_count: ",
                          val_class_counts[0])
                    print("original_val/class_ones_count: ",
                          val_class_counts[1])

                    # # Calculate class weights
                    # train_class_weights = 1. / \
                    #     torch.tensor(train_class_counts, dtype=torch.float)
                    # # Map class labels to indices
                    # train_class_to_index = {
                    #     self.class_name[i]: i for i in range(len(self.class_name))}

                    # train_label_indices = [train_class_to_index[label]
                    #                        for label in train_label_data]

                    # # Assign weights to each sample in the validation set
                    # train_weights = train_class_weights[train_label_indices]

                    # Assuming you have WeightedRandomSampler, you can use it like this:
                    self.weighted_sampler_train = WeightedRandomSampler(
                        weights=[(0.8*self.batch_size) // 100,
                                 (0.2*self.batch_size) // 100],
                        num_samples=len(train_label_data),
                        replacement=False
                    )

                    # Calculate class weights
                    # val_class_weights = 1. / \
                    #     torch.tensor(val_class_counts, dtype=torch.float)

                    # Assuming you have WeightedRandomSampler, you can use it like this:
                    self.weighted_sampler_val = WeightedRandomSampler(
                        weights=[(0.8*self.batch_size) // 100,
                                 (0.2*self.batch_size) // 100],
                        num_samples=len(val_label_data),
                        replacement=False
                    )
                    original_train_input_data = train_input_data.tolist()
                    original_train_label_data = train_label_data.tolist()

                    geo_images = [f for f in os.listdir(
                        self.geo_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    color_images = [f for f in os.listdir(
                        self.color_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    combined_train_data = original_train_input_data + geo_images + color_images

                    # Assuming label_data is all 1 for the length of the new images
                    combined_label_data = original_train_label_data + \
                        [1] * (len(geo_images) + len(color_images))
                    combined_train_data = geo_images + color_images

                    # Assuming label_data is all 1 for the length of the new images
                    combined_label_data = [1] * \
                        (len(geo_images) + len(color_images))

                    print("augmented_train/class_ones_count: ",
                          train_class_counts[1] + len(geo_images) + len(color_images))

                    self.data_train = IsbiDataSet(
                        combined_train_data, combined_label_data, self.class_name, len(combined_train_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=True, image_size=self.image_size)

                    self.data_val = IsbiDataSet(
                        val_input_data.tolist(), val_label_data.tolist(), self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=False, image_size=self.image_size)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.balance_data:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                persistent_workers=True
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                persistent_workers=True,
                # sampler=self.weighted_sampler_train
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            # sampler=self.weighted_sampler_val
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def calculate_class_distribution(self):
        # class_distribution = {
        #     label: 0 for item in self.train_label_list for label in item.keys()}
        class_distribution = {item: 0 for item in self.class_name}

        for item in self.train_label_list:
            class_distribution[item['final label']] += 1

        return class_distribution

    def split_balance_datasets(self, unbalance_class_name):
        # List to store the training DataFrames
        train_dfs = []
        val_dfs = []
        # Function to return a random sample of 70% data from each class

        def sample_data(group, ratio):
            return group.sample(frac=ratio)

        # Loop through each class column
        for class_column in self.train_gt_pdf.columns[1:]:
            # Extract the current class label
            current_class = class_column

            # Filter rows for the current class
            class_df = self.train_gt_pdf[self.train_gt_pdf[current_class] == 1.0]

            # Sample 80% of the data for the current class
            sampled_train_df = class_df.groupby(current_class).apply(
                sample_data, ratio=self.training_split)

            train_dfs.append(sampled_train_df)

        train_data = pd.concat(train_dfs)
        remain_df = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image'])]
        # Add the following assertion
        assert len(remain_df) + len(train_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data with original ground truth!"

        val_data = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image'])]

        assert len(train_data) + len(val_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data and val_data with original ground truth!"
        assert len(self.train_gt_pdf['image'].unique()) == len(
            self.train_gt_pdf), "Duplicate instances in original train_data!"
        assert len(train_data['image'].unique()) == len(
            train_data), "Duplicate instances in train_data!"
        # Check if values in the "image" column of val_data are unique
        assert len(val_data['image'].unique()) == len(
            val_data), "Duplicate instances in val_data!"
        # Convert the "image" columns to sets
        train_images_set = set(train_data['image'])
        val_images_set = set(val_data['image'])
        # Check for overlap using set operations
        assert not train_images_set.intersection(
            val_images_set), "Overlap between train_data and val_data!"
        # Check if values in the "image" column of train_data are unique

        return train_data, val_data

    def split_datasets(self):
        # List to store the training DataFrames
        train_dfs = []
        val_dfs = []
        # Function to return a random sample of 70% data from each class

        def sample_data(group, ratio):
            return group.sample(frac=ratio)

        # Loop through each class column
        for class_column in self.class_name:
            # Extract the current class label
            current_class = class_column

            # Filter rows for the current class
            class_df = self.train_gt_pdf[self.train_gt_pdf[current_class] == 1.0]

            # Sample 80% of the data for the current class
            sampled_train_df = class_df.groupby(current_class).apply(
                sample_data, ratio=self.training_split)

            train_dfs.append(sampled_train_df)

        train_data = pd.concat(train_dfs)
        remain_df = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image'])]
        # Add the following assertion
        assert len(remain_df) + len(train_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data with original ground truth!"

        val_data = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image'])]

        assert len(train_data) + len(val_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data and val_data with original ground truth!"
        assert len(self.train_gt_pdf['image'].unique()) == len(
            self.train_gt_pdf), "Duplicate instances in original train_data!"
        assert len(train_data['image'].unique()) == len(
            train_data), "Duplicate instances in train_data!"
        # Check if values in the "image" column of val_data are unique
        assert len(val_data['image'].unique()) == len(
            val_data), "Duplicate instances in val_data!"
        # Convert the "image" columns to sets
        train_images_set = set(train_data['image'])
        val_images_set = set(val_data['image'])
        # Check for overlap using set operations
        assert not train_images_set.intersection(
            val_images_set), "Overlap between train_data and val_data!"
        # Check if values in the "image" column of train_data are unique

        return train_data, val_data


class IsbiDataSet(Dataset):
    def __init__(self, data, label, class_name, data_length, data_dir, train_image_path, is_transform, train_transforms, val_transforms, is_training, image_size):
        super().__init__()
        self.data = data
        self.label = label
        self.class_name = class_name
        self.train_image_path = train_image_path
        self.data_length = data_length
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.is_training = is_training
        self.image_size = image_size

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Load image using self.df['image'][index], assuming 'image' is the column containing image paths
        image = None
        image_name = self.data[index]
        # Check if the image name ends with specific strings
        if ("color_aug") in self.data[index]:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.data_dir,  "color_aug_images", self.data[index] + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed

        elif "horizontal_flip_image" in image_name or "vertical_flip_image" in image_name or "rotated_image" in image_name:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.data_dir,  "geo_aug_images", image_name + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed
        else:
            try:
                # Attempt to open the image with .jpg extension
                image_path = os.path.join(
                    self.data_dir,  self.train_image_path, image_name + ".jpg")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed

            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.data_dir,  self.train_image_path, image_name + ".png")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    image = Image.open(image_path).convert(
                        'RGB')  # Adjust as needed

                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.data_dir,  self.train_image_path, image_name + ".jpeg")
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
                if self.is_training:
                    image = self.train_transforms(image)
                else:
                    image = self.val_transforms(image)
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                image = self.transform(image)

            label = self.label[index]

            # Create a one-hot encoded tensor
            if len(self.class_name) > 2:
                one_hot_encoded = torch.zeros(
                    len(self.class_name), dtype=torch.float32)
                one_hot_encoded[label] = torch.ones(1, dtype=torch.float32)
            else:
                one_hot_encoded = torch.tensor(label, dtype=torch.float32)

            return image, one_hot_encoded
        else:
            return None


if __name__ == "__main__":
    _ = IsbiDataModule()
