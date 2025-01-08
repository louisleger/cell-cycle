PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
from modules.learning.models import *
from modules.learning.train import train_model
from modules.utils import hc
from modules.visualize import plot_loss, plot_normalized_time_error

# from modules.learning.evaluate import Evaluation, get_latent_space, plot_umap
from matplotlib import pyplot as plt
from modules.learning.dycep_vit import DYCEP2
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# enet = EfficientNet()


track_path = PATH + "track_datasets/control_mm/train/images/"
label_path = PATH + "track_datasets/control_mm/train/labels/"
embeddings_path = PATH + "track_datasets/control_mm/train/embeddings/"

# training mamba model
model = DYCEP2()
model.to(DEVICE)

#########
# test on one track
#########

track_name = "0607.1629.npy"
# getting one sequence to check the model

label = torch.tensor(
    np.load(label_path + track_name).reshape(2, -1).T,
    dtype=torch.float32,
)

emb = torch.tensor(
    np.load(embeddings_path + track_name),
    dtype=torch.float32,
)

print(emb.shape)

label, emb = label.to(DEVICE), emb.to(DEVICE)
zz = model.forward(emb[None, :, :])


train_model(
    directory=PATH + "track_datasets/control_mm/",
    model=model,
    use_embeddings=True,
    batch_size=1,
    learning_rate=1e-4,
    slice_p=0.0,
    random_len=True,
    name="DYCEP",
    num_epochs=1,
)
