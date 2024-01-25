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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


class IsbiDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        image_size: int = 224,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 111,
        kfold_seed_list: list = [111, 222, 333, 444, 555],
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

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

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
        self.kfold_seed_list = kfold_seed_list
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
        self.train_gt_path = self.train_gt_path.replace("\\", "/")

        self.binary_unbalance_train_ratio = 110

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
                # choose fold to train on
                kf = StratifiedKFold(n_splits=len(self.kfold_seed_list),
                                     shuffle=True, random_state=self.kfold_seed)
                all_splits = [k for k in kf.split(
                    input_data, labels)]
                k = self.kfold_seed_list.index(self.kfold_seed)
                train_indexes, val_indexes = all_splits[k]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

                train_input_data = input_data[train_indexes].tolist()
                train_label_data = labels[train_indexes].tolist()

                val_input_data = input_data[val_indexes].tolist()
                val_label_data = labels[val_indexes].tolist()

                # Compute class weights for WeightedRandomSampler
                class_weights = compute_class_weight('balanced', classes=np.unique(
                    labels[train_indexes]), y=labels[train_indexes])
                class_weights = torch.FloatTensor(class_weights)

                # Create a WeightedRandomSampler
                self.sampler = WeightedRandomSampler(
                    class_weights, len(class_weights), replacement=True)
                self.data_train = IsbiDataSet(
                    train_input_data, train_label_data, self.class_name, len(train_input_data), self.data_dir, self.train_image_path, self.is_transform, self.transforms)

                self.data_val = IsbiDataSet(
                    val_input_data, val_label_data, self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.transforms)
            else:
                if self.balance_data:
                    input_data = self.train_gt_pdf['Eye ID']
                    labels = self.train_gt_pdf['Final Label']
                    # Choose fold to train on
                    kf = StratifiedKFold(n_splits=len(self.kfold_seed_list),
                                         shuffle=True, random_state=self.kfold_seed)
                    all_splits = [k for k in kf.split(input_data, labels)]
                    k = self.kfold_seed_list.index(self.kfold_seed)
                    train_indexes, val_indexes = all_splits[k]

                    # Count the number of samples in class 1 in the training set
                    train_count_class_1 = (
                        labels.iloc[train_indexes] == 'RG').sum()
                    train_count_class_0 = (
                        labels.iloc[train_indexes] == 'NRG').sum()
                    val_count_class_1 = (
                        labels.iloc[val_indexes] == 'RG').sum()
                    val_count_class_0 = (
                        labels.iloc[val_indexes] == 'NRG').sum()
                    # Remove samples of class 0 until it equals the count of class 1
                    train_class_0_indexes = train_indexes[labels.iloc[train_indexes] == "NRG"]
                    train_class_1_indexes = train_indexes[labels.iloc[train_indexes] == "RG"]

                    np.random.shuffle(train_class_0_indexes)

                    class_0_indexes_to_keep = train_class_0_indexes[: (
                        train_count_class_1 * (self.binary_unbalance_train_ratio / 100))]

                    # Combine class 1 indexes with selected class 0 indexes
                    # Merge class_1_indexes and class_0_indexes_to_keep
                    merged_indexes = np.concatenate(
                        [train_class_1_indexes, class_0_indexes_to_keep])

                    # Shuffle the merged indexes
                    np.random.shuffle(merged_indexes)
                    # Assuming you want equal samples from each class in a batch
                    # train_num_samples_per_class = train_count_class_1 // 2
                    # self.train_balanced_batch_sampler = BatchSampler(
                    #     WeightedRandomSampler(
                    #         torch.ones(len(train_class_1_indexes)), train_num_samples_per_class, replacement=True
                    #     ),
                    #     WeightedRandomSampler(
                    #         torch.ones(len(train_class_1_indexes)), train_num_samples_per_class, replacement=True
                    #     ),
                    #     batch_size=self.batch_size,
                    #     drop_last=True,
                    # )
                    # val_num_samples_per_class = val_count_class_1 // 2
                    # self.val_balanced_batch_sampler = BatchSampler(
                    #     WeightedRandomSampler(
                    #         torch.ones(len(train_class_1_indexes)), val_num_samples_per_class, replacement=True
                    #     ),
                    #     WeightedRandomSampler(
                    #         torch.ones(len(train_class_1_indexes)), val_num_samples_per_class, replacement=True
                    #     ),
                    #     batch_size=self.batch_size,
                    #     drop_last=True,
                    # )
                    train_input_data = input_data[merged_indexes].tolist()
                    train_label_data = labels[merged_indexes].tolist()

                    val_input_data = input_data[val_indexes].tolist()
                    val_label_data = labels[val_indexes].tolist()

                    self.data_train = IsbiDataSet(
                        train_input_data, train_label_data, self.class_name, len(train_input_data), self.data_dir, self.train_image_path, self.is_transform, self.transforms)

                    self.data_val = IsbiDataSet(
                        val_input_data, val_label_data, self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        sampler = WeightedRandomSampler(
            weights=[0.5, 0.5], num_samples=self.batch_size)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
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
    def __init__(self, data, label, class_name, data_length, data_dir, train_image_path, is_transform, transform):
        super().__init__()
        self.data = data
        self.label = label
        self.class_name = class_name
        self.train_image_path = train_image_path
        self.data_length = data_length
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.transform = transform

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Load image using self.df['image'][index], assuming 'image' is the column containing image paths
        image = None
        try:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.data_dir,  self.train_image_path, self.data[index] + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed

        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.data_dir,  self.train_image_path, self.data[index] + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed

            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.data_dir,  self.train_image_path, self.data[index] + ".jpeg")
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

            # Extract class labels, assuming 'MEL', 'NV', etc., are columns in your CSV file
            label = self.label[index]
            gt = self.class_name.index(label)
            # Create a one-hot encoded tensor
            if len(self.class_name) > 2:
                one_hot_encoded = torch.zeros(
                    len(self.class_name), dtype=torch.float32)
                one_hot_encoded[gt] = torch.ones(1, dtype=torch.float32)
            else:
                one_hot_encoded = torch.tensor(gt, dtype=torch.float32)

            return image, one_hot_encoded
        else:
            return None


if __name__ == "__main__":
    _ = IsbiDataModule()
