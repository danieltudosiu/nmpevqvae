from torch import cat
from torch import tensor
from torch import reshape
from torch.nn import PairwiseDistance

from robust_loss.robust_loss_pytorch.adaptive import AdaptiveVolumeLossFunction


class AdaptiveLoss(object):
    def __init__(self, image_shape, image_device, lambda_reconstruction=1):
        super(AdaptiveLoss).__init__()

        self.av_loss = lambda x: AdaptiveVolumeLossFunction(
            image_size=image_shape, device=image_device
        ).lossfun(x.permute(0, 2, 3, 4, 1))
        self.lambda_reconstruction = lambda_reconstruction

    def __call__(self, network_outputs):
        originals = network_outputs[("org")]
        reconstructions = network_outputs[("rec")]

        loss_total = None
        summaries = {}

        adaptive_loss = (
            self.av_loss(originals - reconstructions) * self.lambda_reconstruction
        ).mean()

        summaries[("summaries", "scalar", "Adaptive-Loss")] = adaptive_loss
        summaries[
            ("summaries", "scalar", "Lambda-Reconstruction")
        ] = self.lambda_reconstruction

        for key in network_outputs:
            if len(key) == 2:
                if key[1] == "ql":
                    if loss_total is None:
                        loss_total = network_outputs[key]
                    else:
                        loss_total += network_outputs[key]

                    summaries[
                        (
                            "summaries",
                            "scalar",
                            "L2-VQ_" + str.upper(str(key[0])) + "-Loss",
                        )
                    ] = network_outputs[key]

        loss_total += adaptive_loss

        summaries[("summaries", "scalar", "Total_Loss")] = loss_total

        return loss_total, summaries

    def set_lambda_reconstruction(self, lambda_reconstruction):
        self.lambda_reconstruction = lambda_reconstruction
        return self.lambda_reconstruction


class BaurLoss(object):
    def __init__(self, lambda_reconstruction=1):
        super(BaurLoss).__init__()

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gdl = 0

        self.l1_loss = lambda x, y: PairwiseDistance(p=1)(
            x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        ).sum()
        self.l2_loss = lambda x, y: PairwiseDistance(p=2)(
            x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        ).sum()

    def __call__(self, network_outputs):
        originals = network_outputs[("org")]
        reconstructions = network_outputs[("rec")]

        summaries = {}

        loss_total = None

        l1_reconstruction = (
            self.l1_loss(originals, reconstructions) * self.lambda_reconstruction
        )
        l2_reconstruction = (
            self.l2_loss(originals, reconstructions) * self.lambda_reconstruction
        )

        summaries[("summaries", "scalar", "L1-Reconstruction-Loss")] = l1_reconstruction
        summaries[("summaries", "scalar", "L2-Reconstruction-Loss")] = l2_reconstruction
        summaries[
            ("summaries", "scalar", "Lambda-Reconstruction")
        ] = self.lambda_reconstruction

        originals_gradients = self.__image_gradients(originals)
        reconstructions_gradients = self.__image_gradients(reconstructions)

        l1_gdl = (
            self.l1_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l1_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l1_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        l2_gdl = (
            self.l2_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l2_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l2_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        summaries[("summaries", "scalar", "L1-Image_Gradient-Loss")] = l1_gdl
        summaries[("summaries", "scalar", "L2-Image_Gradient-Loss")] = l2_gdl
        summaries[("summaries", "scalar", "Lambda-Image_Gradient")] = self.lambda_gdl

        for key in network_outputs:
            if len(key) == 2:
                if key[1] == "ql":
                    if loss_total is None:
                        loss_total = network_outputs[key]
                    else:
                        loss_total += network_outputs[key]

                    summaries[
                        (
                            "summaries",
                            "scalar",
                            "L2-VQ_" + str.upper(str(key[0])) + "-Loss",
                        )
                    ] = network_outputs[key]

        loss_total += l1_reconstruction + l2_reconstruction + l1_gdl + l2_gdl

        summaries[("summaries", "scalar", "Total_Loss")] = loss_total

        return loss_total, summaries

    def set_lambda_reconstruction(self, lambda_reconstruction):
        self.lambda_reconstruction = lambda_reconstruction
        return self.lambda_reconstruction

    def set_lambda_gdl(self, lambda_gdl):
        self.lambda_gdl = lambda_gdl
        return self.lambda_gdl

    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, features, depth, height, width = input_shape

        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        dzz = tensor(()).new_zeros(
            (batch_size, features, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = cat([dz, dzz], 2)
        dz = reshape(dz, input_shape)

        dyz = tensor(()).new_zeros(
            (batch_size, features, depth, 1, width), device=image.device, dtype=dy.dtype
        )
        dy = cat([dy, dyz], 3)
        dy = reshape(dy, input_shape)

        dxz = tensor(()).new_zeros(
            (batch_size, features, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = cat([dx, dxz], 4)
        dx = reshape(dx, input_shape)

        return dx, dy, dz
