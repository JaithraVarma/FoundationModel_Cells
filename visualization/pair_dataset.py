import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from csv import DictReader

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_pairs_for_chunk(args):
    chunk, class_indices, score, shuffle_pairs = args
    np.random.seed()  # Reset the random seed for each process
    indices2 = []

    for i in chunk:
        class1 = class_indices['classes'][i]
        score1 = score[i]

        if np.random.rand() < 0.5:
            idx2 = np.random.choice(class_indices[class1])
            score2 = score[idx2]
            while np.abs(int(score1) - int(score2)) > 5:
                idx2 = np.random.choice(class_indices[class1])
                score2 = score[idx2]
        else:
            class2 = np.random.choice(list(set(class_indices.keys()) - {class1, 'classes'}))
            idx2 = np.random.choice(class_indices[class2])
            score2 = score[idx2]
            while not (10 < np.abs(int(score1) - int(score2)) < 300):
                class2 = np.random.choice(list(set(class_indices.keys()) - {class1, 'classes'}))
                idx2 = np.random.choice(class_indices[class2])
                score2 = score[idx2]
        indices2.append(idx2)

    return indices2


class PairDataset(Dataset):

    def getdata(self, filename):
        data = defaultdict(list)  # dictionary to store data with headers as keys
        with open(filename, 'r') as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                for key in csv_reader.fieldnames:
                    data[key].append(row[key])

        return data

    def __init__(self, path, shuffle_pairs=True, augment=False, num_workers=16):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''
        logger.info(f"Gathering data from {path}")
        data = self.getdata(path)

        self.path = data['path']
        self.classes = data['class']
        self.score = data['imagescore']

        self.feed_shape = [3, 224, 224]
        self.shuffle_pairs = shuffle_pairs

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(int(self.feed_shape[1] * 1.1), int(self.feed_shape[2] * 1.1))),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
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

        self.create_pairs(num_workers)
        self.pair_idxs = list(zip(self.indices1, self.indices2))

        logger.info(f"Length of indices1: {len(self.indices1)}")
        logger.info(f"Length of indices2: {len(self.indices2)}")

    def create_pairs(self, num_workers=16):
        logger.info(f"Creating pairs for {len(self.path)} images")
        class_indices = defaultdict(list)
        for i, class_name in enumerate(self.classes):
            class_indices[class_name].append(i)
        class_indices['classes'] = self.classes  # Add classes for multiprocessing

        self.indices1 = np.arange(len(self.path))
        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            np.random.seed(8)

        chunks = np.array_split(self.indices1, num_workers)
        pool = multiprocessing.Pool(num_workers)
        results = pool.map(create_pairs_for_chunk,
                           [(chunk, class_indices, self.score, self.shuffle_pairs) for chunk in chunks])
        pool.close()
        pool.join()

        self.indices2 = np.concatenate(results)

    def __getitem__(self, idx):
        idx1 = self.pair_idxs[idx][0]
        idx2 = self.pair_idxs[idx][1]

        image_path1 = self.path[idx1]
        image_path2 = self.path[idx2]

        class1 = self.classes[idx1]
        class2 = self.classes[idx2]

        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1).float()
            image2 = self.transform(image2).float()

        return (image1, image2), torch.FloatTensor([class1 == class2]), (class1, class2), (image_path1, image_path2)

    def __len__(self):
        return len(self.path)