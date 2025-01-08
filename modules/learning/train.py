import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import numpy as np
import argparse
from modules.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import R2Score
from modules.learning.dycep import DYCEP
from modules.learning.dycep_vit import DYCEP2
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
import datetime, time


load_dotenv()
print(os.getenv("PYTORCH_CUDA_ALLOC_CONF"))  # Prints "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"


def format_date_yyyymmddhhmm(datetime_obj):
    """
    Formats a datetime object into a string with the format YYYYMMDDHHMM.
    """
    return datetime_obj.strftime("%Y_%m_%d_%H_%M")


# PyTorch Dataset Class for loading cell track datasets
class track_dataset(Dataset):
    def __init__(
        self,
        directory,
        use_embeddings=False,
        img_channels=[0],
        slice_p=0,
        slice_len=1,
        random_len=False,
    ):
        # Save track directory and list of cell names
        self.img_directory = directory + "images/"
        self.label_directory = directory + "labels/"
        self.embeddings_directory = directory + "embeddings/"
        self.use_embeddings = use_embeddings

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


def train_model(
    directory,
    model,
    name,
    num_epochs=30,
    use_embeddings=False,
    img_channels=[0, 0, 0],
    batch_size=1,
    learning_rate=1e-4,
    weights_dir="weights/",
    slice_p=0,
    slice_len=1,
    random_len=False,
):
    """
    Trains the model on the dataset located at directory.
    Saves the model weights and training configuration in weights_dir.

    Parameters:
    directory: str, path to the dataset
    model: nn.Module, the model to train
    name: str, name of the model
    num_epochs: int, number of epochs to train
    use_embeddings: bool, whether to use embeddings or images
    img_channels: list, channels to use from the images
    batch_size: int, batch size
    learning_rate: float, learning rate
    weights_dir: str, path to save the weights
    slice_p: float, probability to slice the track
    slice_len: int, minimum length of the slice
    random_len: bool, whether to use random length slices
    """
    now = datetime.datetime.now()
    formatted_datetime = format_date_yyyymmddhhmm(now)
    # Setting up a configuration json file
    config = {
        "number": formatted_datetime,
        "path": weights_dir,
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
        "temporal_encoder": model.temporal_encoder.__class__.__name__,
        "partial_track_prob": slice_p,
        "training_time": 0,
    }
    # Setting up data loaders
    train_Dataset = track_dataset(
        directory + "train/",
        img_channels=img_channels,
        slice_p=slice_p,
        slice_len=slice_len,
        random_len=random_len,
        use_embeddings=use_embeddings,
    )

    test_Dataset = track_dataset(
        directory + "test/",
        img_channels=img_channels,
        slice_p=slice_p,
        slice_len=slice_len,
        random_len=random_len,
        use_embeddings=use_embeddings,
    )

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(
        train_Dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_Dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Setting up objective function and optimizer
    # loss_function = nn.MSELoss()  # We could try different things for this
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    score = R2Score().to(DEVICE)
    model.to(DEVICE)

    start_time = time.time()

    try:
        for epoch in range(num_epochs):
            model.train()
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = {"train": [], "test": [], "train_R2": [], "test_R2": []}
            i = 0
            # Iterate through elements of training dataset
            for name, inputs, labels, mask in tqdm(train_loader):
                # Attach inputs and labels to device

                inputs, labels, mask = (
                    inputs.to(DEVICE),
                    labels.to(DEVICE),
                    mask.to(DEVICE),
                )

                # Reset optimizer gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)

                # Ignore padded values
                outputs, labels = outputs[mask], labels[mask]

                # Compute Loss and update weights with gradient descent
                loss = ((outputs - labels) ** 2).mean()
                loss.backward()
                optimizer.step()
                # Save Batch Loss and Score
                score.update(outputs, labels)
                running_loss["train"].append(loss.item())
                running_loss["train_R2"].append(score.compute().item())

            # Validate with test set
            model.eval()
            with torch.no_grad():
                for name, inputs, labels, mask in tqdm(test_loader):
                    inputs, labels, mask = (
                        inputs.to(DEVICE),
                        labels.to(DEVICE),
                        mask.to(DEVICE),
                    )
                    outputs = model(inputs)

                    outputs, labels = outputs[mask], labels[mask]

                    loss = ((outputs - labels) ** 2).mean()
                    score.update(outputs, labels)
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
        end_time = time.time()
        config["training_time"] = end_time - start_time
        save_config(config, model, weights_dir)
    except KeyboardInterrupt:
        save_config(config, model, weights_dir)
    return "Done!"


def get_model(args):

    if args.temporal_encoder == "mamba":
        from modules.learning.time_encoders.mamba import Mamba, MambaConfig

        temporal_encoder = Mamba(
            MambaConfig(
                args.temporal_encoder_dim, args.temporal_encoder_layers, d_state=16
            )
        )
    elif args.temporal_encoder == "transformer":
        temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.temporal_encoder_dim,
                nhead=8,
                dim_feedforward=args.temporal_encoder_dim * 2,
                batch_first=True,
            ),
            num_layers=args.temporal_encoder_layers - 1,
        )
    elif args.temporal_encoder == "LSTM":
        temporal_encoder = nn.LSTM(
            input_size=args.temporal_encoder_dim,
            hidden_size=args.temporal_encoder_dim,
            num_layers=args.temporal_encoder_layers,
            batch_first=True,
            bidirectional=False,
        )

    model = DYCEP2(
        # cnn_in_channels=len(args.in_channels),
        temporal_encoder=temporal_encoder,
        vanilla_weights_dir=args.vanilla_weights_dir,
        # freeze=args.freeze,
    )
    return model


if __name__ == "__main__":
    # Sanity checks on shapes and stuff
    parser = argparse.ArgumentParser(description="Train a cell cycle model")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="track_datasets/control_mm/",
        help="dataset path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/datasets/",
    )
    parser.add_argument("--spatial_encoder", type=str, default="custom_cnn")
    parser.add_argument("--temporal_encoder", type=str, default="mamba")
    parser.add_argument("--physics_informed", type=bool, default=True)
    parser.add_argument("--use_embeddings", type=bool, default=False)
    parser.add_argument("--name", type=str, default="DYCEP2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=[1])
    parser.add_argument("--temporal_encoder_dim", type=int, default=256)
    parser.add_argument("--temporal_encoder_layers", type=int, default=6)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--slice_p", type=float, default=0)
    # partial tracks training arguments
    parser.add_argument("--slice_len", type=int, default=1)
    parser.add_argument("--random_len", type=bool, default=False)
    parser.add_argument(
        "--vanilla_weights_dir", type=str, default="vanilla/coef_fourier.npy"
    )

    args = parser.parse_args()

    model = get_model(args)

    train_model(
        DATA_PATH + args.dataset_path,
        model=model,
        img_channels=args.in_channels,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        name=args.name,
        num_epochs=args.epochs,
        slice_p=args.slice_p,
        slice_len=args.slice_len,
        random_len=args.random_len,
        use_embeddings=args.use_embeddings,
    )
    print("All Done!")
