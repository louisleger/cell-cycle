from transformers import AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoFeatureExtractor, SwinModel
import numpy as np

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
import sys

sys.path.append("..")


#############
# loading data
#############


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

track_path = PATH + "track_datasets/control_mm/train/images/"
embedding_path = PATH + "track_datasets/control_mm/train/embeddings/"
in_channels = [1, 1, 1]
bf_channel = [1]

import os

cells = os.listdir(track_path)[:]

# tracks = []
# for cell in tqdm(cells):
#     track = np.load(track_path + cell, allow_pickle=True)[:, bf_channel, :, :]
#     # min max all images in track
#     track = (track - track.min(axis=(2, 3), keepdims=True)) / (
#         track.max(axis=(2, 3), keepdims=True) - track.min(axis=(2, 3), keepdims=True)
#     )
#     # repeat 3 times the channel
#     track = np.repeat(track, 3, axis=1)
#     inputs = feature_extractor(images=track, return_tensors="pt")


# there is no padding, so it doenst work
class CellDataset(Dataset):
    def __init__(self, cells, track_path, bf_channel):
        self.cells = cells
        self.track_path = track_path
        self.bf_channel = bf_channel

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cell = self.cells[idx]
        track = np.load(self.track_path + cell, allow_pickle=True)[
            :, self.bf_channel, :, :
        ]
        # Min-max normalize
        track = (track - track.min(axis=(1, 2), keepdims=True)) / (
            track.max(axis=(1, 2), keepdims=True)
            - track.min(axis=(1, 2), keepdims=True)
        )
        # Repeat 3 times the channel
        track = np.repeat(track, 3, axis=1)
        return track


dataset = CellDataset(cells, track_path, bf_channel)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# track = np.load(track_path + cells[0], allow_pickle=True)[:, bf_channel, :, :]
# # Min-max normalize
# track = (track - track.min(axis=(1, 2), keepdims=True)) / (
#     track.max(axis=(1, 2), keepdims=True) - track.min(axis=(1, 2), keepdims=True)
# )
# # Repeat 3 times the channel
# track = np.repeat(track, 3, axis=1)
# track.shape

#############
# model
#############


for batch in tqdm(dataloader):
    inputs = feature_extractor(images=batch, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
