from glob import glob
from os import remove as os_remove

from torch import save as torch_save
from nibabel import Nifti1Image
from utils import visualisation_normalisation


def calculate_gdl_lambda(engine):
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


def log_summaries(engine, logger):
    output = engine.state.output
    step = engine.state.iteration

    for key, value in output.items():
        if key[0] == "summaries":
            summary_type = key[1]
            summary_tag = key[2]
            summary_value = value

            if summary_type == "image3":
                logger.add_video(
                    tag=summary_tag,
                    vid_tensor=visualisation_normalisation(summary_value.permute(0, 2, 1, 3, 4)),
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


def save_output(engine, output_path):
    output = engine.state.output
    file_names = engine.state.batch["t1"]["stem"]
    affine_transformations = engine.state.batch["t1"]["affine"]

    for idx in range(len(file_names)):
        nii = Nifti1Image(
            output[("rec")][idx, 0, ...].detach().cpu().numpy(),
            affine_transformations[idx].detach().cpu().numpy(),
        )
        nii.header["qform_code"] = 1
        nii.header["sform_code"] = 0
        nii.to_filename(output_path + file_names[idx])


def save_checkpoint(engine, model, optimizer, lr_scheduler, amp, no_checkpoints, checkpoint_directory):
    step = engine.state.iteration

    checkpoints = [
        int(e.split("/")[-1].split("_")[-1].split(".")[0])
        for e in glob(checkpoint_directory + "*.pth")
    ]
    checkpoints.sort()

    if len(checkpoints) > no_checkpoints:
        os_remove(checkpoint_directory + "checkpoint_" + str(checkpoints[0]) + ".pth")

    torch_save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'engine': engine.state_dict(),
            'amp': amp.state_dict()
        },
        checkpoint_directory + "checkpoint_" + str(step) + ".pth"
    )