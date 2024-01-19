from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import os
import pandas as pd
import numpy as np
from PIL import Image


class IsicDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        is_transform=True
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

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.training_split = train_val_test_split[0]
        self.validation_split = train_val_test_split[1]
        self.test_split = train_val_test_split[2]
        self.is_transform = is_transform
        self.train_image_path = os.path.join(
            self.data_dir, "ISIC-2019/ISIC_2019_Training_Input")
        self.test_image_path = os.path.join(
            self.data_dir, "ISIC-2019/ISIC_2019_Test_Input")
        self.train_gt_path = os.path.join(
            self.data_dir, "ISIC-2019/ISIC_2019_Training_GroundTruth.csv")
        self.train_metadata_gt_path = os.path.join(
            self.data_dir, "ISIC-2019/ISIC_2019_training_Metadata.csv")

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 8

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Load the CSV file into a pandas DataFrame
        self.train_gt_pdf = pd.read_csv(self.train_gt_path)
        self.train_gt_pdf.drop(columns=['UNK'], inplace=True)
        self.train_image_name = self.train_gt_pdf["image"]
        self.train_label_list = self.train_gt_pdf.iloc[:, 1:].apply(
            lambda row: {col.lower(): row[col] for col in self.train_gt_pdf.columns[1:]}, axis=1).tolist()

        self.class_distribution = self.calculate_class_distribution()
        # Calculate the number of samples for each split
        self.total_samples = int(sum(self.class_distribution.values()))
        # Create indices array
        all_indices = np.arange(self.total_samples)

        # Get the sampled indices from the train_sampler

        self.train_size = int(self.training_split * self.total_samples)
        self.train_indices = np.array(all_indices[:self.train_size])
        self.valid_size = int(self.validation_split * (self.total_samples))
        self.test_size = self.total_samples - self.train_size - self.valid_size

        # Assign weights to each sample based on its class frequency
        class_frequencies = [self.class_distribution[class_name]
                             for class_name in self.class_distribution]
        self.weights = [1.0 / class_freq for class_freq in class_frequencies]
        self.data_train_df, self.data_val_df, self.data_test_df = self.split_datasets()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # if stage == "fit":
        #     self.data_train = IsicDataSet(self.data_train_df,
        #                                   len(self.data_train_df), self.data_dir, self.is_transform, self.transforms)
        # if stage == "validate":
        #     self.data_val = IsicDataSet(self.data_val_df,
        #                                 len(self.data_val_df), self.data_dir, self.is_transform, self.transforms)
        # if stage == "test":
        #     self.data_test = IsicDataSet(self.data_test_df,
        #                                  len(self.data_test_df), self.data_dir, self.is_transform, self.transforms)
        self.data_train_df.reset_index(drop=True, inplace=True)
        self.data_val_df.reset_index(drop=True, inplace=True)
        self.data_test_df.reset_index(drop=True, inplace=True)

        self.data_train = IsicDataSet(self.data_train_df,
                                      len(self.data_train_df), self.data_dir, self.is_transform, self.transforms)
        self.data_val = IsicDataSet(self.data_val_df,
                                    len(self.data_val_df), self.data_dir, self.is_transform, self.transforms)
        self.data_test = IsicDataSet(self.data_test_df,
                                     len(self.data_test_df), self.data_dir, self.is_transform, self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
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
            shuffle=False,
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
        class_distribution = {
            label: 0 for item in self.train_label_list for label in item.keys()}

        for item in self.train_label_list:
            for label, count in item.items():
                class_distribution[label] += count

        return class_distribution

    def split_datasets(self):
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

            # Sample 70% of the data for the current class
            sampled_train_df = class_df.groupby(current_class).apply(
                sample_data, ratio=self.training_split)

            train_dfs.append(sampled_train_df)

        train_data = pd.concat(train_dfs)
        remain_df = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image'])]
        # Add the following assertion
        assert len(remain_df) + len(train_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data with original ground truth!"
        # Loop through each class column
        for class_column in remain_df.columns[1:]:
            # Extract the current class label
            current_class = class_column

            # Filter rows for the current class
            class_df = remain_df[remain_df[current_class] == 1.0]

            # Sample 70% of the data for the current class
            sampled_valid_df = class_df.groupby(current_class).apply(
                sample_data, ratio=0.5)

            val_dfs.append(sampled_valid_df)

        val_data = pd.concat(val_dfs)

        test_data = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image']) & ~self.train_gt_pdf['image'].isin(val_data['image'])]

        assert len(train_data) + len(val_data) + len(test_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data and val_data with original ground truth!"
        assert len(self.train_gt_pdf['image'].unique()) == len(
            self.train_gt_pdf), "Duplicate instances in train_data!"
        assert len(train_data['image'].unique()) == len(
            train_data), "Duplicate instances in train_data!"

        # Check if values in the "image" column of val_data are unique
        assert len(val_data['image'].unique()) == len(
            val_data), "Duplicate instances in val_data!"

        # Check if values in the "image" column of test_data are unique
        assert len(test_data['image'].unique()) == len(
            test_data), "Duplicate instances in test_data!"

        # Convert the "image" columns to sets
        train_images_set = set(train_data['image'])
        val_images_set = set(val_data['image'])
        test_images_set = set(test_data['image'])

        # Check for overlap using set operations
        assert not train_images_set.intersection(
            val_images_set), "Overlap between train_data and val_data!"
        assert not train_images_set.intersection(
            test_images_set), "Overlap between train_data and test_data!"
        assert not val_images_set.intersection(
            test_images_set), "Overlap between val_data and test_data!"

        # Check if values in the "image" column of train_data are unique

        return train_data, val_data, test_data


class IsicDataSet(Dataset):
    def __init__(self, gt_dataframe, data_length, data_dir, is_transform, transform):
        super().__init__()
        self.gt_dataframe = gt_dataframe
        self.data_length = data_length
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.transform = transform

    def __len__(self):
        return len(self.gt_dataframe)

    def __getitem__(self, index):
        df = self.gt_dataframe
        # Load image using self.df['image'][index], assuming 'image' is the column containing image paths
        image_path = os.path.join(
            self.data_dir, "ISIC-2019/ISIC_2019_Training_Input/", df['image'][index] + ".jpg")
        # Replacing backslashes with forward slashes
        image_path = image_path.replace("\\", "/")
        image = Image.open(image_path).convert('RGB')  # Adjust as needed
        # Apply transformations if specified
        if self.is_transform:
            image = self.transform(image)

        # Extract class labels, assuming 'MEL', 'NV', etc., are columns in your CSV file
        labels = df.iloc[index, 1:].values.astype(float)
        labels = torch.tensor(labels)
        return image, labels


if __name__ == "__main__":
    _ = IsicDataModule()
