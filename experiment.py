from os import listdir
from os import fsdecode
from os.path import join

from os.path import exists

from pathlib import Path

from time import time

from glob import glob

from csv import writer

from math import ceil

from torch import no_grad
from torch import device as torch_device
from torch import load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd.profiler import profile
from torch.autograd.profiler import emit_nvtx

from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.contrib.handlers import ProgressBar

from torchio import Image
from torchio import ImagesDataset
from torchio import INTENSITY
from torchio import Subject

from vq_vae import VectorQuantizedVAE
from lr_schedulers import CosineDecayRestarts
from parser import cmd_parser
from losses import BaurLoss
from losses import AdaptiveLoss
from utils import min_max_scale


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


def unsupervised_training_function(
    model, loss_function, optimizer, lr_scheduler, device
):
    def process_function(engine, batch):
        # Setting the model for training
        model.train()

        # Cleaning the past gradients
        optimizer.zero_grad()

        # Fetching the data from the batch and moving to device (TorchIO specific)
        input = batch["t1"]["data"].to(device)

        # Getting model outputs
        output = model(input)

        if isinstance(loss_function, BaurLoss):
            loss_function.set_lambda_gdl(engine.state.lambda_gdl)

        # Calculating the loss
        loss, loss_ar = loss_function(output)

        # Storing the loss and its additional returns
        output["loss"] = loss.item()
        output["loss_ar"] = loss_ar

        # Calculating the gradients
        loss.backward()

        # Applying the gradients
        optimizer.step()

        # Step the learning rate scheduler
        lr_scheduler.step()

        return output

    return Engine(process_function)


def unsupervised_testing_function(model, loss_function, device):
    def process_function(engine, batch):
        # Setting the model for evaluation
        model.eval()

        with no_grad():
            # Fetching the data from the batch and moving to device (TorchIO specific)
            input = batch["t1"]["data"].to(device)

            # Getting model outputs
            output = model(input)

            # Calculating the loss
            loss, loss_ar = loss_function(output)

            # Storing the loss and its additional returns
            output["loss"] = loss.item()
            output["loss_ar"] = loss_ar

            # Saving everything the in the engine
            engine.state.output = output

        return output

    return Engine(process_function)


def calcualte_gdl_lambda(engine):
    if engine.state.iteration < engine.hparams["zero_image_gradient_loss"]:
        engine.state.lambda_gdl = 0.0
    else:
        engine.state.lambda_gdl = min(
            1
            * (
                (engine.state.iteration - engine.hparams["zero_image_gradient_loss"])
                / engine.hparams["one_image_gradient_loss"]
            ),
            engine.hparams["max_image_gradient_loss"],
        )


def log_every_x_steps(engine, logger):
    output = engine.state.output
    step = engine.state.iteration

    for summaries in [output["summaries"], output["loss_ar"]["summaries"]]:
        for summary_type, type_summaries in summaries.items():
            for summary_tag, summary_value in type_summaries.items():
                if summary_type == "image3":
                    logger.add_video(
                        tag=summary_tag,
                        vid_tensor=min_max_scale(summary_value.permute(0, 2, 1, 3, 4)),
                        global_step=step,
                        fps=12,
                    )
                elif summary_type == "scalar":
                    logger.add_scalar(
                        tag=summary_tag, scalar_value=summary_value, global_step=step
                    )
                elif summary_type == "histogram":
                    logger.add_histogram(
                        tag=summary_tag, values=summary_value, global_step=step
                    )


arguments = [
    "-trdd",
    "/raid/danieltudosiu/datasets/neuro_morphology/healthy/train_192",
    "-tsdd",
    "/raid/danieltudosiu/datasets/neuro_morphology/healthy/test_192",
    "-pd",
    "/raid/danieltudosiu/projects/pytorch_nmcvqvae/",
    "-en",
    "profiling",
    "-d",
    "1",
    "-m",
    "Training",
    "-si",
    "0",
    "-e",
    "20000",
    "-le",
    "1000",  # 10000 originally
    "-ce",
    "1000",  # 10000 originally
    "-cl",
    "100",
    "-b",
    "3",
    "-lr",
    "0.0001",
    "-l",
    "baur",
    "-rl",
    "1.0",
    "-zgdl",
    "100000",
    "-ogdl",
    "10000",
    "-mgdl",
    "5",
    "-fds",
    "6480",
    "-a",
    "0.0000001",
    "-tm",
    "1.25",
    "-mm",
    "0.95",
]

args = cmd_parser.parse_args(arguments)
hparams = vars(args)

experiment_directory = args.project_directory + args.experiment_name

if exists(experiment_directory) and args.starting_iteration == 0:
    raise ValueError(
        "Directory already exists! Please delete it or rename the experiment!"
    )

checkpoint_directory = experiment_directory + "/checkpoints/"
logs_directory = experiment_directory + "/logs/"
outputs_directory = experiment_directory + "/outputs/"

Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
Path(logs_directory).mkdir(parents=True, exist_ok=True)
Path(outputs_directory).mkdir(parents=True, exist_ok=True)

csv_writer = writer(open(experiment_directory + "/hparams.csv", "w"))
for arg, val in hparams.items():
    csv_writer.writerow([arg, val])

device = (
    torch_device("cpu") if args.device == "" else torch_device("cuda:" + args.device)
)

summary_writer = SummaryWriter(logs_directory)

if args.mode == "Training":
    data_loader = get_data_loader(args.training_data_directory, args.batch_size)
else:
    data_loader = get_data_loader(args.testing_data_directory, args.batch_size)

model = VectorQuantizedVAE()
model = model.to(device)

if args.loss == "baur":
    loss_function = BaurLoss(
        lambda_reconstruction=args.reconstruction_lambda,
        # This will be automatically calculated before each iteration
        lambda_gdl=0,
    )
else:
    loss_function = AdaptiveLoss(
        image_shape=iter(data_loader).__next__()["t1"]["data"].shape[1:],
        image_device=device,
        lambda_reconstruction=args.reconstruction_lambda,
    )

optimizer = Adam(params=model.parameters(), lr=args.learning_rate)

lr_scheduler = CosineDecayRestarts(
    optimizer=optimizer,
    first_decay_steps=args.first_decay_steps,
    alpha=args.alpha,
    t_mul=args.t_mul,
    m_mul=args.m_mul,
)

if args.mode == "Training":
    engine = unsupervised_training_function(
        model, loss_function, optimizer, lr_scheduler, device
    )
else:
    engine = unsupervised_testing_function(model, loss_function, device)

engine.hparams = hparams

checkpoint_state = {
    "model": model,
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "trainer": engine,
}


if args.starting_iteration != 0:
    checkpoints = [
        int(e.split("/")[-1].split("_")[-1].split(".")[0])
        for e in glob(checkpoint_directory + "*.pth")
    ]
    checkpoints.sort()

    if args.starting_iteration == -1:
        checkpoint_iteration = checkpoints[-1]
    elif args.starting_iteration in checkpoints:
        checkpoint_iteration = args.starting_iteration
    else:
        raise ValueError("Checkpoint iteration does not exist!")

    checkpoint = load(
        checkpoint_directory + "checkpoint_" + str(checkpoint_iteration) + ".pth"
    )
    Checkpoint.load_objects(to_load=checkpoint_state, checkpoint=checkpoint)

if args.mode == "Training":
    engine.add_event_handler(Events.ITERATION_STARTED, calcualte_gdl_lambda)
    engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.log_every),
        log_every_x_steps,
        summary_writer,
    )
    engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.checkpoint_every),
        Checkpoint(
            checkpoint_state,
            DiskSaver(dirname=checkpoint_directory, create_dir=True),
            n_saved=args.checkpoint_last,
        ),
    )
else:
    # TODO: Hook to save the outputs
    pass

pbar = ProgressBar()
pbar.attach(engine, output_transform=lambda output: {"loss": output["loss"]})

e = engine.run(data=data_loader, max_epochs=args.epochs)
