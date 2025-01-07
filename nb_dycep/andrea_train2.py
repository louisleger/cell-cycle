PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
from modules.learning.models import *
from modules.learning.train import train_model
from modules.utils import hc
from modules.visualize import plot_loss, plot_normalized_time_error

# from modules.learning.evaluate import Evaluation, get_latent_space, plot_umap
from matplotlib import pyplot as plt
from modules.learning.dycep import DYCEP
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# enet = EfficientNet()
model = DYCEP()
model.to(DEVICE)

track_path = PATH + "track_datasets/control_mm/train/images/"
label_path = PATH + "track_datasets/control_mm/train/labels/"


#########
# test on one track
#########

track_name = "0607.1629.npy"
# getting one sequence to check the model
imgs = torch.tensor(
    np.load(track_path + track_name, allow_pickle=True),
    dtype=torch.float32,
)[:, [1], :, :]


label = torch.tensor(
    np.load(label_path + track_name).reshape(2, -1).T,
    dtype=torch.float32,
)

imgs, label = imgs.to(DEVICE), label.to(DEVICE)

zz = model.forward(imgs[None, :, :, :])

plt.plot(zz.detach().cpu().numpy().squeeze())
print(hc(model), "Parameters")

model.vanilla_weights.shape
taus = torch.linspace(0, 1, 1000)[None, :].to(DEVICE)
taus.device

van = model.vanilla_fn(taus).squeeze().detach().cpu().numpy()



#############
# actual training
#############



train_model(
    directory=PATH + "track_datasets/control_mm/",
    model=model,
    # pass [1] for only BF channel, or [1,1,1] for 3 times the same channel
    # or [0,1,2] for all channels
    img_channels=[0],
    batch_size=1,
    learning_rate=1e-4,
    slice_p=0,
    name="DYCEP",
    num_epochs=10,
)

# config-10.json
plot_loss(["weights/config-14.json"])

# load dict in .json file
with open("weights/config-14.json", "r") as file:
    config = json.load(file)

config


from modules.learning.evaluate import Evaluation


model.load_state_dict(torch.load("weights/model-14.pt"))

eval = Evaluation()
eval.fit(
    PATH + "track_datasets/control_mm/test/", model, img_channels=[0], smoothing=True
)

print(eval.summary()), eval.visualize_predicted_tracks

eval.prediction_df.y_hat[0].shape

for i in range(100):
    track = eval.prediction_df.y_hat[i]
    tau = np.linspace(0, 1, track.shape[0])
    plt.plot(tau, track[:, 0], alpha=0.2)


for i in range(100):
    track = eval.prediction_df.y_hat[i]
    tau = np.linspace(0, 1, track.shape[0])
    plt.plot(tau, track[:, 1], alpha=0.2)
