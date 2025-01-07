from transformers import AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoFeatureExtractor, SwinModel
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model.to(DEVICE)

PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
import sys

sys.path.append("..")


#############
# loading data
#############


fucci_path = PATH + "track_datasets/control_mm/train/labels/"
track_path = PATH + "track_datasets/control_mm/train/images/"
embedding_path = PATH + "track_datasets/control_mm/train/embeddings/"
in_channels = [1, 1, 1]
bf_channel = 1

import os

cells = os.listdir(track_path)[:]
cells_done = os.listdir(embedding_path)[:]

# check intersection and difference of cells
intersection = set(cells) & set(cells_done)
difference = set(cells) - set(cells_done)

print(f"Cells done: {len(intersection)}")
print(f"Cells to do: {len(difference)}")
print(f"Total cells: {len(cells)} sum = {len(intersection) + len(difference)}")

difference = list(difference)

#############
# model
#############


for cell in tqdm(difference):
    # Load the track
    track = np.load(track_path + cell, allow_pickle=True)[:, bf_channel, :, :]
    if len(track.shape) != 3:
        continue

    # Min-max normalize all frames
    track = (track - track.min(axis=(1, 2), keepdims=True)) / (
        track.max(axis=(1, 2), keepdims=True) - track.min(axis=(1, 2), keepdims=True)
    )

    # Repeat channel to convert grayscale to RGB
    track = np.repeat(track[:, None, :, :], 3, axis=1)  # Shape: (S, 3, H, W)

    # Process each frame in the track
    track_embeddings = np.zeros((len(track), 768))
    for i, frame in enumerate(track):
        inputs = feature_extractor(images=frame, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (E,)
            track_embeddings[i, :] = embedding.cpu().numpy()

    # Save track embeddings as a tensor of shape (S, E)
    np.save(embedding_path + cell, track_embeddings)
