import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import logging

logger = logging.getLogger(__name__)
# log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

logger.setLevel(logging.INFO)


class ImageDataset(Dataset):

    def __init__(self, path, shuffle_pairs=True, augment=False, num_workers=16):
        """
        Create an iterable dataset from a directory containing sub-directories of
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        """
        logger.info(f"Gathering data from {path}")
        self.data = pd.read_csv(path)
        print(f"Number of pairs before filtering: {len(self.data)}")
        # add "column" pat number
        self.data['pat_no'] = self.data['class'].apply(lambda x: x.split("_")[1])
        # filter pat_no  9346 or 9633 or 9502
        self.data = self.data[self.data['pat_no'].isin(['9346', '9633', '9502'])]
        print(f"Number of pairs after filtering: {len(self.data)}")

        self.feed_shape = [3, 224, 224]

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(int(self.feed_shape[1] * 1.1), int(self.feed_shape[2] * 1.1))),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(int(self.feed_shape[1] * 1.1), int(self.feed_shape[2] * 1.1))),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])

    def __getitem__(self, idx):
        idx_data = self.data.iloc[idx]

        image_path = idx_data['path']

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image).float()
        label = idx_data['class']

        return image, label, image_path

    def __len__(self):
        return len(self.data)
