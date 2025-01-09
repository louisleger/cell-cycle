from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import torch
import os
import sys
from torch.nn.utils.rnn import pad_sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
import numpy as np


class track_dataset(Dataset):
    def __init__(
        self,
        directory,
        use_embeddings=False,
        img_channels=[0],
        slice_p=0,
        slice_len=1,
        random_len=False,
        load_in_memory=False
    ):
        # Save track directory and list of cell names
        self.img_directory = directory + "images/"
        self.label_directory = directory + "labels/"
        self.embeddings_directory = directory + "embeddings/"
        self.use_embeddings = use_embeddings
        self.load_in_memory = load_in_memory

        # Save selection of specific channels for images from [PC, BF, H2B]
        self.img_channels = img_channels

        if use_embeddings:
            self.cells = [
                cell
                for cell in os.listdir(self.embeddings_directory)
                if cell != "nan.npy"
            ]
        else:
            self.cells = [
                cell for cell in os.listdir(self.img_directory) if cell != "nan.npy"
            ]

        # Load tracks in memory
        if load_in_memory: 
            self.cells = [[self.cells, torch.tensor(np.load(self.img_directory + cell_name, allow_pickle=True), dtype=torch.float32)[:, self.img_channels, :, :],
                            torch.tensor(np.load(self.label_directory + cell_name).reshape(2, -1).T,dtype=torch.float32)]
                            for cell_name in self.cells]
            

        # Probability to create an x-y slice of the track
        self.slice_p = slice_p
        self.slice_len = slice_len
        self.random_len = random_len

    def intensity_shift(self, imgs):
        return imgs - torch.randn(1)

    def random_slice(self, imgs, labels):
        slice_len = self.slice_len
        if self.random_len:
            slice_len = np.random.randint(self.slice_len, labels.shape[0] - 1)
        x = np.random.randint(0, labels.shape[0] - slice_len)
        return (
            imgs[x : x + slice_len],
            labels[x : x + slice_len],
        )  # minimum 2h of images

    def __getitem__(self, idx):

        if self.use_embeddings:
            # Load embeddings S x 768 and labels, S x F (fucci = 2)
            embeddings = torch.tensor(
                np.load(self.embeddings_directory + self.cells[idx], allow_pickle=True),
                dtype=torch.float32,
            )
            labels = torch.tensor(
                np.load(self.label_directory + self.cells[idx]).reshape(2, -1).T,
                dtype=torch.float32,
            )

            if np.random.rand() < self.slice_p and labels.shape[0] > self.slice_len:
                embeddings, labels = self.random_slice(embeddings, labels)
            return self.cells[idx], embeddings, labels

        elif self.load_in_memory:
            name, imgs, labels = tuple(self.cells[idx])
            
            # Augmentation
            imgs = self.intensity_shift(imgs)

            if np.random.rand() < self.slice_p and labels.shape[0] > self.slice_len:
                imgs, labels = self.random_slice(imgs, labels)

            return name, imgs, labels
        else: 

            # Load images and labels, S (time) x C x H x W
            imgs = torch.tensor(
                np.load(self.img_directory + self.cells[idx], allow_pickle=True),
                dtype=torch.float32,
            )[:, self.img_channels, :, :]
            # Load labels, T x F (fucci = 2)
            labels = torch.tensor(
                np.load(self.label_directory + self.cells[idx]).reshape(2, -1).T,
                dtype=torch.float32,
            )
            # Augmentation
            imgs = self.intensity_shift(imgs)

            if np.random.rand() < self.slice_p and labels.shape[0] > self.slice_len:
                imgs, labels = self.random_slice(imgs, labels)
            return self.cells[idx], imgs, labels

    def __len__(self):
        return len(self.cells)


def collate_fn(batch):
    # batch is a list of tuples: (name, imgs, labels)
    names = [track[0] for track in batch]
    imgs = [track[1] for track in batch]
    labels = [track[2] for track in batch]

    imgs_padded = pad_sequence(imgs, batch_first=True, padding_value=-100)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return names, imgs_padded, labels_padded, labels_padded != -1


