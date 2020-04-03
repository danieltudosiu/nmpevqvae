from argparse import ArgumentParser

cmd_parser = ArgumentParser()

# ========================= File system parameters =========================

cmd_parser.add_argument(
    "--training_data_directory",
    help="The location of the training data",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "--testing_data_directory",
    help="The location of the testing data",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "--project_directory",
    help="The parent folder for the experiment. A folder with the experiment name will be created within.",
    action="store",
    type=str,
)

# ========================= Experiment parameters =========================

cmd_parser.add_argument(
    "--experiment_name",
    help="The name of the experiment. A folder with this name will be created in project_directory",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "--mode",
    help="Which mode the network will be used it, Training or Inference.",
    action="store",
    choices=["Training", "Inference"],
    type=str,
    default="Training",
)

cmd_parser.add_argument(
    "--starting_iteration",
    help=(
        "Iteration at which we start/load the mode, Must be either 0 for a new experiment, "
        + "the iteration of a checkpoint or -1 for the last checkpoint"
    ),
    action="store",
    type=int,
    default=0,
)

# ========================= Hardware parameters =========================

cmd_parser.add_argument(
    "--device", help="Which GPU is going to be used are going to use.", action="store", type=int
)

cmd_parser.add_argument(
    "--deterministic",
    help="To use deterministic behaviour or not. A heavy performance penalty might be incurred if determinism is preferred.",
    action="store_true",
    default=False,
)

# ========================= Monitoring parameters =========================

cmd_parser.add_argument(
    "--log_every",
    help="The period of steps after which we log.",
    action="store",
    type=int,
    default=10000,
)

cmd_parser.add_argument(
    "--checkpoint_every",
    help="The period of steps after which we checkpoint.",
    action="store",
    type=int,
    default=10000,
)

cmd_parser.add_argument(
    "--checkpoint_last",
    help="How many checkpoints to save. It will delete older checkpoints as new ones are being created.",
    action="store",
    type=int,
    default=10,
)

# ========================= Training parameters =========================

cmd_parser.add_argument(
    "--epochs",
    help="The maximum number of epochs to train.",
    action="store",
    type=int,
)

cmd_parser.add_argument(
    "--batch_size",
    help="The number of samples in a batch.",
    action="store",
    type=int,
    default=3,
)

cmd_parser.add_argument(
    "--learning_rate",
    help="The initial upper bound of the learning rate",
    action="store",
    type=float,
    default=0.0001,
)

cmd_parser.add_argument(
    "--loss",
    help="Which loss to be used for the experiment. It can either be Baur loss or Adaptive loss.",
    action="store",
    type=str,
    choices=["Baur", "Adaptive"],
    default="Adaptive",
)

cmd_parser.add_argument(
    "--reconstruction_lambda",
    help="The lambda for the reconstruction loss",
    action="store",
    type=float,
    default=1.0,
)

# ========================= Baur Loss parameters =========================

cmd_parser.add_argument(
    "--zero_image_gradient_loss",
    help="The number of iterations for which the image gradient loss alpha will be 0",
    action="store",
    type=int,
    default=100000,
)

cmd_parser.add_argument(
    "--one_image_gradient_loss",
    help="The number of iterations for across which the image gradient loss alpha will go from 0 to 1",
    action="store",
    type=int,
    default=10000,
)

cmd_parser.add_argument(
    "--max_image_gradient_loss",
    help="The maximum image gradient loss alpha",
    action="store",
    type=int,
    default=5,
)

# ========================= SGDR parameters =========================

cmd_parser.add_argument(
    "--first_decay_steps",
    help="The initial T_0 across which Adam's upper bound will be lowered from the initial learning rate to the learning rate * alpha",
    action="store",
    type=int,
    default=6480,
)

cmd_parser.add_argument(
    "--alpha",
    help="The learning rate lower bound as determined by current learning rate * alpha",
    action="store",
    type=float,
    default=0.0000001,
)

cmd_parser.add_argument(
    "--t_mul",
    help="The amount of increase from one cycle to another for the period",
    action="store",
    type=float,
    default=1.25,
)

cmd_parser.add_argument(
    "--m_mul",
    help="The amount of decay for the upper bound of the learning rate",
    action="store",
    type=float,
    default=0.95,
)
