from argparse import ArgumentParser

cmd_parser = ArgumentParser()

cmd_parser.add_argument(
    "-trdd",
    "--training_data_directory",
    help="The location of the training data",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "-tsdd",
    "--testing_data_directory",
    help="The location of the testing data",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "-pd",
    "--project_directory",
    help="The parent folder for the experiment. A folder with the experiment name will be created.",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "-d",
    "--device",
    help='Which devices to be used. The list must be either empty ""  for CPU or an integet "0" for GPU 0',
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "-e",
    "--epochs",
    help="The maximum number of epochs to train",
    action="store",
    type=int,
)

cmd_parser.add_argument(
    "-m",
    "--mode",
    help="Training or Inference",
    action="store",
    choices=["Training", "Inference"],
    type=str,
)

cmd_parser.add_argument(
    "-si",
    "--starting_iteration",
    help="Iteration at which we start. Must be either 0 for a new experiment, -1 for the last checkpoint, or the iteration of a checkpoint.",
    action="store",
    type=int,
)

cmd_parser.add_argument(
    "-en",
    "--experiment_name",
    help="The name of the experiment. It is going to be appended at the end of the checkpoint_saving_directory and log_saving_directory. Also, the current date and time are being appended.",
    action="store",
    type=str,
)

cmd_parser.add_argument(
    "-le",
    "--log_every",
    help="The period of steps after which we log",
    action="store",
    type=int,
    default=1000,
)

cmd_parser.add_argument(
    "-ce",
    "--checkpoint_every",
    help="The period of steps after which we checkpoint",
    action="store",
    type=int,
    default=1000,
)

cmd_parser.add_argument(
    "-cl",
    "--checkpoint_last",
    help="How many checkpoints to save",
    action="store",
    type=int,
    default=10,
)

cmd_parser.add_argument(
    "-b",
    "--batch_size",
    help="The number of elements in a batch.",
    action="store",
    type=int,
    default=2,
)

cmd_parser.add_argument(
    "-lr",
    "--learning_rate",
    help="The initial upper bound of the learning rate",
    action="store",
    type=float,
    default=0.0001,
)

cmd_parser.add_argument(
    "-l",
    "--loss",
    help="Which loss to be used for the experiment. It can either be Baur loss or Adaptive loss.",
    action="store",
    type=str,
    choices=["baur", "adaptive"],
    default="adaptive",
)


cmd_parser.add_argument(
    "-rl",
    "--reconstruction_lambda",
    help="The lambda for the reconstruction loss",
    action="store",
    type=float,
    default=1.0,
)

cmd_parser.add_argument(
    "-zgdl",
    "--zero_image_gradient_loss",
    help="The number of iterations for which the image gradient loss alpha will be 0",
    action="store",
    type=int,
    default=100000,
)

cmd_parser.add_argument(
    "-ogdl",
    "--one_image_gradient_loss",
    help="The number of iterations for across which the image gradient loss alpha will go from 0 to 1",
    action="store",
    type=int,
    default=10000,
)

cmd_parser.add_argument(
    "-mgdl",
    "--max_image_gradient_loss",
    help="The maximum image gradient loss alpha",
    action="store",
    type=int,
    default=5,
)

cmd_parser.add_argument(
    "-fds",
    "--first_decay_steps",
    help="The initial T_0 across which Adam's upper bound will be lowered from the initial learning rate to the learning rate * alpha",
    action="store",
    type=int,
    default=6480,
)

cmd_parser.add_argument(
    "-a",
    "--alpha",
    help="The learning rate lower bound as determined by current learning rate * alpha",
    action="store",
    type=float,
    default=0.0000001,
)

cmd_parser.add_argument(
    "-tm",
    "--t_mul",
    help="The amount of increase from one cycle to another for the period",
    action="store",
    type=float,
    default=1.25,
)

cmd_parser.add_argument(
    "-mm",
    "--m_mul",
    help="The amount of decay for the upper bound of the learning rate",
    action="store",
    type=float,
    default=0.95,
)
