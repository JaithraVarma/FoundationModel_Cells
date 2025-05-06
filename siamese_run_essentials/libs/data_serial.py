import os
import glob
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import csv


class Dataset(torch.utils.data.IterableDataset):

    def getdata(self, filename):
        data = {}  # dictionary to store data with headers as keys
        # filename = filename[3:]
        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            for row in csv_reader:
                for i in range(len(row)):
                    key = header[i]
                    if key not in data:
                        data[key] = []
                    data[key].append(row[i])
        return data

    def __init__(self, path, shuffle_pairs=True, augment=False):
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
        data = self.getdata(path)

        self.path = data['path']
        self.score = data['imagescore']

        self.feed_shape = [3, 256, 256]
        self.crop_shape = [3, 224, 224]
        self.shuffle_pairs = shuffle_pairs

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(int(self.crop_shape[1] * 1.1), int(self.crop_shape[2] * 1.1))),
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

        self.create_pairs()

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.image_paths = self.path
        self.score = self.score
        self.image_classes = []
        self.class_indices = {}

        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            # saved seed is 8
            np.random.seed(8)

        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        for i in self.indices1:
            class1 = self.image_classes[i]
            score1 = self.score[i]

            # Select an image of the same or different class based on the similarity score difference
            if np.random.rand() < 0.5:
                # Select an image of the same class
                idx2 = np.random.choice(self.class_indices[class1])
                score2 = self.score[idx2]
                while np.abs(int(score1) - int(score2)) > 6:
                    idx2 = np.random.choice(self.class_indices[class1])
                    score2 = self.score[idx2]
            else:
                # Select an image of a different class
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
                idx2 = np.random.choice(self.class_indices[class2])
                score2 = self.score[idx2]
                # while not (10 < np.abs(int(score1) - int(score2)) < 400):
                #     class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
                #     idx2 = np.random.choice(self.class_indices[class2])
                #     score2 = self.score[idx2]
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):

            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            dirname, filename = os.path.split(image_path2)
            file_num = int(filename.split('.')[0])
            new_file_num_1 = file_num - random.randint(0, 2)
            new_file_num_2 = file_num - random.randint(0, 2)
            new_image_path_1 = os.path.join(dirname, str(new_file_num_1) + '.jpg')
            new_image_path_2 = os.path.join(dirname, str(new_file_num_2) + '.jpg')
            image3 = Image.open(new_image_path_1).convert("RGB")
            image4 = Image.open(new_image_path_2).convert("RGB")




            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()
                image3 = self.transform(image3).float()
                image4 = self.transform(image4).float()


            yield (image1, image2 ,image3,image4), torch.FloatTensor([class1 == class2]), (class1, class2)

    def __len__(self):
        return len(self.image_paths)
