import glob
import csv
import pathlib
import os

import torch
import numpy as np
import torchio


TRAINING = "Training"
INFERENCE = "Inference"


def visualisation_normalisation(input_tensor):
    input_tensor_min, _ = input_tensor.view(input_tensor.shape[0], -1).min(1)
    input_tensor_min = input_tensor_min[
        (...,) + (None,) * (len(input_tensor.shape) - len(input_tensor_min.shape))
    ]

    input_tensor = input_tensor - input_tensor_min

    input_tensor_max, _ = input_tensor.view(input_tensor.shape[0], -1).max(1)
    input_tensor_max = input_tensor_max[
        (...,) + (None,) * (len(input_tensor.shape) - len(input_tensor_max.shape))
    ]

    input_tensor = input_tensor / input_tensor_max

    return input_tensor


def load_state(
    checkpoint_directory,
    starting_iteration,
    mode,
    model,
    optimizer,
    lr_scheduler,
    amp,
    engine,
):
    if starting_iteration != 0:
        checkpoints = [
            int(e.split("/")[-1].split("_")[-1].split(".")[0])
            for e in glob.glob(checkpoint_directory + "*.pth")
        ]

        checkpoints.sort()

        if starting_iteration == -1:
            checkpoint_iteration = checkpoints[-1]
        elif starting_iteration in checkpoints:
            checkpoint_iteration = starting_iteration
        else:
            raise ValueError("Checkpoint iteration does not exist!")

        checkpoint = torch.load(
            checkpoint_directory + "checkpoint_" + str(checkpoint_iteration) + ".pth"
        )

        model = model.load_state_dict(checkpoint["model"])
        optimizer = optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler = lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        amp = amp.load_state_dict(checkpoint["amp"])

        if mode == TRAINING:
            engine = engine.load_state_dict(checkpoint["engine"])

    elif starting_iteration == 0 and mode == "Inference":
        raise ValueError(
            "You need to specify a non-zero starting iteration for the inference model to be loaded. "
            "Either -1 for the last one or a specific one."
        )

    return model, optimizer, lr_scheduler, amp, engine


def save_params(mode, experiment_directory, params):
    if mode == TRAINING:
        with open(experiment_directory + "/params.csv", "w") as params_file:
            csv_writer = csv.writer(params_file)

            for arg, val in params.items():
                csv_writer.writerow([arg, val])

            params_file.flush()


def set_deterministic(is_deterministic):
    # As per PyTorch documentation for reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html#cudnn

    torch.manual_seed(0)
    np.random.seed(0)

    if is_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def setup_directories(project_directory, experiment_name, starting_iteration):
    experiment_directory = project_directory + experiment_name

    if os.path.exists(experiment_directory) and starting_iteration == 0:
        raise ValueError(
            "Directory already exists! Please delete it or rename the experiment!"
        )

    checkpoint_directory = experiment_directory + "/checkpoints/"
    logs_directory = experiment_directory + "/logs/"
    outputs_directory = experiment_directory + "/outputs/"

    pathlib.Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logs_directory).mkdir(parents=True, exist_ok=True)
    pathlib.Path(outputs_directory).mkdir(parents=True, exist_ok=True)

    return experiment_directory, checkpoint_directory, logs_directory, outputs_directory


def get_data_loader(data_path, batch_size):
    # Particular way of loading the nii files as per TorchIO documentation
    # https://torchio.readthedocs.io/data/images.html
    subjects = []

    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        if filename.endswith(".nii.gz"):
            subjects.append(
                torchio.Subject(
                    torchio.Image(
                        "t1", os.path.join(data_path, filename), torchio.INTENSITY
                    )
                )
            )

    data_loader = torch.utils.data.DataLoader(
        torchio.ImagesDataset(subjects),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return data_loader


def get_torch_device(device):
    if device == "":
        raise ValueError("You have not passed a GPU index.")
    else:
        try:
            print(int(device))
            torch.cuda.set_device(int(device))
            torch_device = torch.device("cuda")
            print(torch_device)
        except ValueError:
            raise ValueError(
                "The device you have passed is not the index of a single GPU. Please pass and integer index of a single GPU."
            )
    return torch_device
