import torch
import ignite

from torch.utils.tensorboard import SummaryWriter
from ignite.contrib.handlers import ProgressBar
from torchio import DATA
from apex import amp
from apex.optimizers import FusedAdam

import parser
import handlers
import utils
import losses
import lr_schedulers
import network


def unsupervised_training_function(
    model, loss_function, optimizer, lr_scheduler, device
):
    def process_function(engine, batch):
        model.train()

        optimizer.zero_grad()

        # Fetching the data from the batch and moving to device (TorchIO specific)
        input = batch["T1"][DATA].to(device)

        output = model(input)

        if isinstance(loss_function, losses.BaurLoss):
            loss_function.set_lambda_gdl(engine.state.lambda_gdl)

        loss, loss_summaries = loss_function(output)

        output[("loss")] = loss.item()
        output.update(loss_summaries)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        lr_scheduler.step()

        return output

    return ignite.engine.Engine(process_function)


def unsupervised_inference_function(model, loss_function, device):
    def process_function(engine, batch):
        model.eval()

        with torch.no_grad():
            input = batch["T1"][DATA].to(device)

            output = model(input)

            loss, loss_summaries = loss_function(output)

            output[("loss")] = loss.item()
            output.update(loss_summaries)

            engine.state.output = output

        return output

    return ignite.engine.Engine(process_function)


args = parser.cmd_parser.parse_args(
    [
        "--training_data_directory",
        "/home/danieltudosiu/storage/datasets/neuro_morphology/healthy/train_192/",
        "--testing_data_directory",
        "/home/danieltudosiu/storage/datasets/neuro_morphology/healthy/test_192/",
        "--project_directory",
        "/home/danieltudosiu/storage/projects/nmpevqvae/",
        "--experiment_name",
        "adaptive",
        "--device",
        "1",
        "--mode",
        "Training",
        "--starting_iteration",
        "0",
        "--epochs",
        "20000",
        "--log_every",
        "10000",
        "--checkpoint_every",
        "10000",
        "--checkpoint_last",
        "5",
        "--batch_size",
        "2",
        "--learning_rate",
        "0.0001",
        "--loss",
        "Adaptive",
        "--reconstruction_lambda",
        "1.0",
        "--zero_image_gradient_loss",
        "100000",
        "--one_image_gradient_loss",
        "10000",
        "--max_image_gradient_loss",
        "5",
        "--first_decay_steps",
        "6480",
        "--alpha",
        "0.0000001",
        "--t_mul",
        "1.25",
        "--m_mul",
        "0.95",
    ]
)
params = vars(args)

experiment_directory, checkpoint_directory, logs_directory, outputs_directory = utils.setup_directories(
    project_directory=args.project_directory,
    experiment_name=args.experiment_name,
    starting_iteration=args.starting_iteration,
)

utils.set_deterministic(is_deterministic=args.deterministic)

utils.save_params(
    mode=args.mode, experiment_directory=experiment_directory, params=params
)

device = utils.get_torch_device(args.device)

data_loader = utils.get_data_loader(
    data_path=args.training_data_directory
    if args.mode == utils.TRAINING
    else args.testing_data_directory,
    batch_size=args.batch_size,
)

model = network.VectorQuantizedVAE()
model = model.to(device)

if args.loss == "Baur":
    loss_function = losses.BaurLoss(lambda_reconstruction=args.reconstruction_lambda)
else:
    image_shape = iter(data_loader).__next__()["T1"][DATA].shape[1:]
    image_shape = image_shape[1:] + (image_shape[0],)

    loss_function = losses.AdaptiveLoss(
        image_shape=image_shape,
        image_device=int(args.device),
        lambda_reconstruction=args.reconstruction_lambda,
    )

optimizer = FusedAdam(params=model.parameters(), lr=args.learning_rate)

model, optimizer = amp.initialize(
    models=model,
    optimizers=optimizer,
    # !! Pay at your own risk with the opt_level, wasted more than a week trying to use O1 !!
    opt_level="O0",
)

lr_scheduler = lr_schedulers.CosineDecayRestarts(
    optimizer=optimizer,
    first_decay_steps=args.first_decay_steps,
    alpha=args.alpha,
    t_mul=args.t_mul,
    m_mul=args.m_mul,
)

if args.mode == utils.TRAINING:
    engine = unsupervised_training_function(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
    )
else:
    engine = unsupervised_inference_function(
        model=model,
        loss_function=loss_function,
        device=device
    )

engine.hparams = params

checkpoint_state = utils.load_state(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    engine=engine,
    amp=amp,
    checkpoint_directory=checkpoint_directory,
    starting_iteration=args.starting_iteration,
    mode=args.mode,
)

if args.mode == utils.TRAINING:
    engine.add_event_handler(
        ignite.engine.Events.ITERATION_STARTED,
        handlers.calculate_gdl_lambda
    )
    engine.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED(every=args.log_every),
        handlers.log_summaries,
        torch.utils.tensorboard.SummaryWriter(logs_directory),
    )
    engine.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED(every=args.checkpoint_every),
        handlers.save_checkpoint,
        model,
        optimizer,
        lr_scheduler,
        amp,
        args.checkpoint_last,
        checkpoint_directory,
    )
else:
    engine.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED,
        handlers.save_output,
        outputs_directory,
    )

pbar = ProgressBar()
pbar.attach(engine, output_transform=lambda output: {"loss": output[("loss")]})

e = engine.run(data=data_loader, max_epochs=args.epochs if args.mode == utils.TRAINING else 1)
