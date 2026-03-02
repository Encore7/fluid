import os
import random
from typing import Tuple, Union

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms

from src.scripts.helper import metadata, save_metadata


class DataLoader:
    def __init__(
        self,
        dataset_input_feature: str,
        dataset_target_feature: str = "label",
        dataset_name: str = "mnist",
    ):
        self.dataset_name = dataset_name
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.pytorch_transforms = self.pytorch_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _apply_transforms(self, batch):
        if batch.get(self.dataset_input_feature):
            transformed_images = []
            for img in batch[self.dataset_input_feature]:
                if isinstance(img, list):
                    transformed_images.append(torch.tensor(img, dtype=torch.float32))
                else:
                    transformed_images.append(self.pytorch_transforms(img))
            batch[self.dataset_input_feature] = transformed_images
        return batch

    def _load_partition(self, num_clients: int, alpha: float):
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=self.dataset_target_feature,
            alpha=alpha,
            self_balancing=True,
        )
        return FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

    def _get_dataset_metadata(self, fds: FederatedDataset, num_clients: int):
        # Take one full partition to extract dataset-wide metadata
        sample_partition = fds.load_partition(0)
        sample_example = sample_partition[0]

        # Get image shape and channels
        image = sample_example[self.dataset_input_feature]
        print(f"Sample image type: {type(image)}")

        if hasattr(image, "shape"):
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
        elif hasattr(image, "size"):  # PIL Image
            width, height = image.size
            # Infer channels from mode
            mode_to_channels = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            channels = mode_to_channels.get(image.mode, 1)
        else:
            raise ValueError("Cannot determine image dimensions from sample.")

        # Compute number of classes
        all_labels = []
        for client_id in range(num_clients):
            partition = fds.load_partition(client_id)
            all_labels.extend(partition[self.dataset_target_feature])

        num_classes = len(set(all_labels))

        # Save metadata
        return {
            "num_classes": num_classes,
            "image_width": width,
            "image_height": height,
            "num_channels": channels,
        }

    def save_datasets_to_disk(
        self,
        num_clients: int,
        alpha: float,
        dataset_folder_path: str,
    ):
        fds = self._load_partition(num_clients, alpha)

        # Get dataset metadata
        save_metadata(self._get_dataset_metadata(fds, num_clients))

        for client_id in range(num_clients):
            client_dataset_folder_path = os.path.join(
                dataset_folder_path, f"client_{client_id}"
            )
            os.makedirs(client_dataset_folder_path, exist_ok=True)

            partition = fds.load_partition(client_id)

            labels = partition[self.dataset_target_feature]
            # Compute the unique classes and their counts
            unique_classes, counts = np.unique(labels, return_counts=True)

            # Filter out classes with only one row
            classes_to_keep = set(unique_classes[counts > 1])
            partition = partition.filter(
                lambda example, keep=classes_to_keep: example[
                    self.dataset_target_feature
                ]
                in keep,
                load_from_cache_file=False,
            )

            partition_train_val = partition.train_test_split(
                test_size=0.2, seed=42, stratify_by_column=self.dataset_target_feature
            )

            train_partition = partition_train_val["train"]
            val_partition = partition_train_val["test"]

            train_path = os.path.join(client_dataset_folder_path, "train_data")
            val_path = os.path.join(client_dataset_folder_path, "val_data")

            train_partition.save_to_disk(train_path)
            val_partition.save_to_disk(val_path)

        # Load and save test dataset (common for all clients)
        test_set = load_dataset(self.dataset_name, split="test")
        test_path = os.path.join(dataset_folder_path, "test_data")
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        test_set.save_to_disk(test_path)

    def load_dataset_from_disk(
        self,
        data_type: str,
        client_folder_path,
        batch_size,
        num_of_batches,
        dataset_input_feature,
        dataset_target_feature,
        is_drifted,
        percentage_to_swap,
        abrupt_drift_labels_swap,
        create_fedau_dataloader,
        invert_drift_selection,
    ) -> Union[TorchDataLoader, Tuple[TorchDataLoader, TorchDataLoader]]:
        client_file_path = os.path.join(client_folder_path, data_type)

        client_dataset = load_from_disk(client_file_path).with_transform(
            self._apply_transforms
        )

        num_of_samples = batch_size * num_of_batches

        if num_of_samples > len(client_dataset):
            num_of_samples = len(client_dataset)

        indices = np.random.choice(
            len(client_dataset), size=num_of_samples, replace=False
        ).tolist()

        client_dataset = client_dataset.select(indices)

        if is_drifted or invert_drift_selection:
            swapped_data = []
            if create_fedau_dataloader:
                fedau_dataloader = []

            for sample in client_dataset:
                # Get the sample (assumed to be in the format (image, label))
                image = sample[dataset_input_feature]
                label = sample[dataset_target_feature]
                if create_fedau_dataloader:
                    fedau_label = sample[dataset_target_feature]

                if invert_drift_selection and not is_drifted:
                    percentage_to_swap = 1.0
                # Check each swap rule
                for rule in abrupt_drift_labels_swap:
                    if random.random() < percentage_to_swap:
                        if label == rule["label1"]:
                            label = rule["label2"]

                            if create_fedau_dataloader:
                                fedau_label = random.randint(
                                    0, metadata["num_classes"] - 1
                                )
                        elif label == rule["label2"]:
                            label = rule["label1"]

                            if create_fedau_dataloader:
                                fedau_label = random.randint(
                                    0, metadata["num_classes"] - 1
                                )
                # Append the (possibly modified) sample to our swapped data list
                swapped_data.append(
                    {dataset_input_feature: image, dataset_target_feature: label}
                )

                if create_fedau_dataloader:
                    fedau_dataloader.append(
                        {
                            dataset_input_feature: image,
                            dataset_target_feature: fedau_label,
                        }
                    )

            client_dataset = swapped_data

        data_loader = TorchDataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        if create_fedau_dataloader:
            fedau_dataloader = TorchDataLoader(
                fedau_dataloader,
                batch_size=batch_size,
                shuffle=True,
            )
            return data_loader, fedau_dataloader

        return data_loader

    def load_test_dataset_from_disk(
        self,
        dataset_folder_path: str,
        batch_size: int,
    ) -> TorchDataLoader:
        test_path = os.path.join(dataset_folder_path, "test_data")

        test_dataset = load_from_disk(test_path).with_transform(self._apply_transforms)

        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        return test_loader
