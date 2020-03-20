from os import fsdecode
from os import listdir
from os.path import join

import torch
from torch.autograd.profiler import emit_nvtx
from torch.cuda.profiler import profile
from torch.cuda import synchronize
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchio import INTENSITY
from torchio import Image
from torchio import ImagesDataset
from torchio import Subject

from apex import pyprof

from losses import BaurLoss
from vq_vae import VectorQuantizedVAE


def get_data_loader(data_path, batch_size):
    subjects = []

    for file in listdir(data_path):
        filename = fsdecode(file)
        if filename.endswith(".nii.gz"):
            subjects.append(Subject(Image("t1", join(data_path, filename), INTENSITY)))

    data_loader = DataLoader(
        ImagesDataset(subjects), batch_size=batch_size, shuffle=True, drop_last=True
    )

    return data_loader


torch.backends.cudnn.benchmark = True
pyprof.nvtx.init()

device = "cuda:1"
warm_up_iterations = 20
benchmark_iterations = 100

data_loader = get_data_loader(
    "/raid/danieltudosiu/datasets/neuro_morphology/healthy/train_192", 3
)
batch = iter(data_loader).__next__()
input = batch["t1"]["data"].to(device)

model = VectorQuantizedVAE()
model = model.to(device)

loss_function = BaurLoss(
    lambda_reconstruction=1,
    # This will be automatically calculated before each iteration
    lambda_gdl=0,
)

optimizer = Adam(params=model.parameters(), lr=0.0001)

model.train()

for i in range(warm_up_iterations):
    optimizer.zero_grad()
    output = model(input)
    loss, loss_ar = loss_function(output)
    loss.backward()
    optimizer.step()

synchronize(device=device)

with profile():
    with emit_nvtx():
        for i in range(benchmark_iterations):
            optimizer.zero_grad()
            output = model(input)
            loss, loss_ar = loss_function(output)
            loss.backward()
            optimizer.step()
