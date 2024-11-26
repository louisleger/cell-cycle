import os
import torch
import numpy as np
from modules.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import R2Score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"


# PyTorch Dataset Class for loading cell track datasets
class track_dataset(Dataset):
    def __init__(
        self, directory, img_channels=[0], slice_p=0, slice_len=1, random_len=False
    ):
        # Save track directory and list of cell names
        self.img_directory = directory + "images/"
        self.label_directory = directory + "labels/"
        self.cells = [
            cell for cell in os.listdir(self.img_directory) if cell != "nan.npy"
        ]

        # Save selection of specific channels for images from [PC, BF, H2B]
        self.img_channels = img_channels
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
        # Load images and labels
        imgs = torch.tensor(
            np.load(self.img_directory + self.cells[idx], allow_pickle=True),
            dtype=torch.float32,
        )[:, self.img_channels, :, :]
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


def train_model(
    directory,
    model,
    name,
    num_epochs=30,
    img_channels=[0, 0, 0],
    batch_size=1,
    learning_rate=1e-4,
    weights_dir="",
    slice_p=0,
    slice_len=1,
    random_len=False,
):
    # Setting up a configuration json file
    config = {
        "number": len(os.listdir(weights_dir + "weights/")),
        "path": weights_dir + "weights/",
        "name": name,
        "model_type": str(type(model)),
        "img_channels": img_channels,
        "batch_size": batch_size,
        "num_epochs": 0,
        "learning_rate": learning_rate,
        "train_loss": [],
        "test_loss": [],
        "train_R2": [],
        "test_R2": [],
    }
    # Setting up data loaders
    train, test = track_dataset(
        directory + "train/", img_channels, slice_p, slice_len, random_len
    ), track_dataset(directory + "test/", img_channels, slice_p, slice_len, random_len)
    train_loader, test_loader = DataLoader(
        train, batch_size, True, num_workers=4, pin_memory=True
    ), DataLoader(test, batch_size, True, num_workers=4, pin_memory=True)

    # Setting up objective function and optimizer
    loss_function = nn.MSELoss()  # We could try different things for this
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    score = R2Score()
    model.to(DEVICE)
    try:
        for epoch in range(num_epochs):
            model.train()
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = {"train": [], "test": [], "train_R2": [], "test_R2": []}
            # Iterate through elements of training dataset
            for name, inputs, labels in tqdm(train_loader):
                # Attach inputs and labels to device
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # Reset optimizer gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                # Compute Loss and update weights with gradient descent
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                # Save Batch Loss and Score
                score.update(outputs.reshape(-1, 2), labels.reshape(-1, 2))
                running_loss["train"].append(loss.item())
                running_loss["train_R2"].append(score.compute().item())

            # Validate with test set
            model.eval()
            with torch.no_grad():
                for name, inputs, labels in tqdm(test_loader):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = loss_function(outputs.reshape(-1, 2), labels.reshape(-1, 2))
                    running_loss["test"].append(loss.item())
                    running_loss["test_R2"].append(score.compute().item())

            # Save Epoch Loss
            config["train_loss"].append(np.mean(running_loss["train"]))
            config["test_loss"].append(np.mean(running_loss["test"]))
            config["train_R2"].append(np.mean(running_loss["train_R2"]))
            config["test_R2"].append(np.mean(running_loss["test_R2"]))
            print(
                f"{4*' '}Train Loss: {config['train_loss'][-1]:.3f}, Test Loss: {config['test_loss'][-1]:.3f}"
            )
            print(
                f"{4*' '}Train R2: {config['train_R2'][-1]:.3f}, Test R2: {config['test_R2'][-1]:.3f}"
            )
            config["num_epochs"] += 1
        save_config(config, model, weights_dir)
    except KeyboardInterrupt:
        save_config(config, model, weights_dir)
    return "Done!"


if __name__ == "__main__":
    # Sanity checks on shapes and stuff
    dataset = track_dataset(DATA_PATH + "track_datasets/healthy/train/")
    print(len(dataset), dataset.__getitem__(0)[0].shape)
