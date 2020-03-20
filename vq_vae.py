import numpy as np
import torch

from torch.autograd.profiler import record_function
from torch.nn import functional as F
from torch import nn


class VectorQuantizedVAE(nn.Module):
    def __init__(self,):
        super(VectorQuantizedVAE, self).__init__()
        self.num_layers = 23

        self.encoder = Encoder(num_layers=self.num_layers)
        self.quantization = Quantization(num_layers=self.num_layers)
        self.decoder = Decoder(num_layers=self.num_layers)

        self.qar = None

    def forward(self, images):
        e6, e4, e2 = self.encoder(images)

        q6, q4, q2, qar = self.quantization(e6, e4, e2)

        out = self.decoder(q6, q4, q2)

        tbr = {
            "org": images,
            "rec": out,
            "q2": {
                "e": e2,
                "q": q2,
                "ql": qar["q2"]["l"],
                "qeo": qar["q2"]["eo"],
                "qei": qar["q2"]["ei"],
                "p": qar["q2"]["p"],
            },
            "q4": {
                "e": e4,
                "q": q4,
                "ql": qar["q4"]["l"],
                "qeo": qar["q4"]["eo"],
                "qei": qar["q4"]["ei"],
                "p": qar["q4"]["p"],
            },
            "q6": {
                "e": e6,
                "q": q6,
                "ql": qar["q6"]["l"],
                "qeo": qar["q6"]["eo"],
                "qei": qar["q6"]["ei"],
                "p": qar["q6"]["p"],
            },
            "summaries": {
                "histogram": {
                    "Q2_Encoding_Indices": qar["q2"]["ei"],
                    "Q4_Encoding_Indices": qar["q4"]["ei"],
                    "Q6_Encoding_Indices": qar["q6"]["ei"],
                },
                "scalar": {
                    "Q2_Perplexity": qar["q2"]["p"].mean(),
                    "Q4_Perplexity": qar["q4"]["p"].mean(),
                    "Q6_Perplexity": qar["q6"]["p"].mean(),
                },
                "image3": {
                    "Originals": images,
                    "Reconstructions": out,
                    "Residuals": images - out,
                },
            },
        }

        return tbr

    def decode(self, q6, q4, q2):
        return self.decoder(q6, q4, q2)


class Encoder(nn.Module):
    def __init__(self, num_layers):
        super(Encoder, self).__init__()

        # -------------------- Level 0 --------------------
        self.e0 = FixupBlock(
            in_channels=1,
            out_channels=4,
            identity_function="level",
            name="Encoder_Level_Level_0_Initial_Depth",
        )
        # -------------------- Level 1 --------------------
        self.e1 = FixupBlock(
            in_channels=4,
            out_channels=8,
            identity_function="downsample",
            name="Encoder_Downsample_Level_1",
        )
        # -------------------- Level 2 --------------------
        self.e2 = FixupBlock(
            in_channels=8,
            out_channels=16,
            identity_function="downsample",
            name="Encoder_Downsample_Level_2",
        )
        # -------------------- Level 3 --------------------
        self.e3 = FixupBlock(
            in_channels=16,
            out_channels=32,
            identity_function="downsample",
            name="Encoder_Downsample_Level_3",
        )
        # -------------------- Level 4 --------------------
        self.e4 = FixupBlock(
            in_channels=32,
            out_channels=64,
            identity_function="downsample",
            name="Encoder_Downsample_Level_4",
        )
        # -------------------- Level 5 --------------------
        self.e5 = FixupBlock(
            in_channels=64,
            out_channels=128,
            identity_function="downsample",
            name="Encoder_Downsample_Level_5",
        )
        # -------------------- Level 6 --------------------
        self.e6 = FixupBlock(
            in_channels=128,
            out_channels=256,
            identity_function="downsample",
            name="Encoder_Downsample_Level_6",
        )

        # ---------- Initialization ----------
        for m in self.modules():
            if isinstance(m, FixupBlock):
                m.initialize_weights(num_layers=num_layers)

    def forward(self, input):
        e2 = self.e2(self.e1(self.e0(input)))
        e4 = self.e4(self.e3(e2))
        e6 = self.e6(self.e5(e4))

        return e6, e4, e2


class Quantization(nn.Module):
    def __init__(self, num_layers):
        super(Quantization, self).__init__()

        # --------------------Level 6 --------------------
        self.q6b = FixupBlock(
            in_channels=256,
            out_channels=32,
            identity_function="level",
            name="Quantization_6_Bottleneck",
        )
        self.q6 = VectorQuantizerEMA(
            num_embeddings=256,
            embedding_dim=32,
            commitment_cost=7,
            decay=0.99,
            epsilon=1e-5,
            name="Quantization_Level_6",
        )
        self.q6c6t5 = FixupBlock(
            in_channels=32,
            out_channels=16,
            identity_function="upsample",
            name="Quantization_6_Upsample_6_to_5",
        )
        self.q6c5t4 = FixupBlock(
            in_channels=16,
            out_channels=8,
            identity_function="upsample",
            name="Quantization_6_Upsample_5_to_4",
        )
        # -------------------- Level 4 --------------------
        self.q4b = FixupBlock(
            in_channels=72,
            out_channels=8,
            identity_function="level",
            name="Quantization_6_Bottleneck",
        )
        self.q4 = VectorQuantizerEMA(
            num_embeddings=256,
            embedding_dim=8,
            commitment_cost=7,
            decay=0.99,
            epsilon=1e-5,
            name="Quantization_Level_4",
        )
        self.q4c4t3 = FixupBlock(
            in_channels=8,
            out_channels=4,
            identity_function="upsample",
            name="Quantization_4_Upsample_4_to_3",
        )
        self.q4c3t2 = FixupBlock(
            in_channels=4,
            out_channels=2,
            identity_function="upsample",
            name="Quantization_4_Upsample_3_to_2",
        )
        # -------------------- Level 2 --------------------
        self.q2b = FixupBlock(
            in_channels=18,
            out_channels=2,
            identity_function="level",
            name="Quantization_2_Bottleneck",
        )
        self.q2 = VectorQuantizerEMA(
            num_embeddings=256,
            embedding_dim=2,
            commitment_cost=7,
            decay=0.99,
            epsilon=1e-5,
            name="Quantization_Level_2",
        )

        # ---------- Initialization ----------
        for m in self.modules():
            if isinstance(m, FixupBlock):
                m.initialize_weights(num_layers=num_layers)

    def forward(self, e6, e4, e2):
        q6l, q6, q6p, q6eo, q6ei = self.q6(self.q6b(e6))
        q6u = self.q6c5t4(self.q6c6t5(q6))
        q4l, q4, q4p, q4eo, q4ei = self.q4(self.q4b(torch.cat((e4, q6u), 1)))
        q4u = self.q4c3t2(self.q4c4t3(q4))
        q2l, q2, q2p, q2eo, q2ei = self.q2(self.q2b(torch.cat((e2, q4u), 1)))

        ar = {
            "q6": {"l": q6l, "eo": q6eo, "ei": q6ei, "p": q6p},
            "q4": {"l": q4l, "eo": q4eo, "ei": q4ei, "p": q6p},
            "q2": {"l": q2l, "eo": q2eo, "ei": q2ei, "p": q6p},
        }

        return q6, q4, q2, ar


class Decoder(nn.Module):
    def __init__(self, num_layers):
        super(Decoder, self).__init__()

        # -------------------- Level 6 --------------------
        self.d6 = FixupBlock(
            in_channels=32,
            out_channels=128,
            identity_function="upsample",
            name="Decoder_Level_6",
        )
        # -------------------- Level 5 --------------------
        self.d5 = FixupBlock(
            in_channels=128,
            out_channels=64,
            identity_function="upsample",
            name="Decoder_Level_5",
        )
        # -------------------- Level 4 --------------------
        self.d4 = FixupBlock(
            in_channels=72,
            out_channels=32,
            identity_function="upsample",
            name="Decoder_Level_4",
        )
        # -------------------- Level 3 --------------------
        self.d3 = FixupBlock(
            in_channels=32,
            out_channels=16,
            identity_function="upsample",
            name="Decoder_Level_3",
        )
        # -------------------- Level 2 --------------------
        self.d2 = FixupBlock(
            in_channels=18,
            out_channels=8,
            identity_function="upsample",
            name="Decoder_Level_2",
        )
        # -------------------- Level 1 --------------------
        # self.d1 = SubPixelConvolution3D(
        #    in_channels=8, out_channels=1, upsample_factor=2, name="Decoder_Level_1"
        # )
        self.d1 = FixupBlock(
            in_channels=8,
            out_channels=4,
            identity_function="upsample",
            name="Decoder_Level_1",
        )

        self.d0 = FixupBlock(
            in_channels=4,
            out_channels=1,
            identity_function="level",
            name="Decoder_Level_0",
        )

        # ---------- Initialization ----------
        for m in self.modules():
            if isinstance(m, FixupBlock):
                m.initialize_weights(num_layers=num_layers)
            # elif isinstance(m, SubPixelConvolution3D):
            #    m.initialize_weights()

    def forward(self, q6, q4, q2):
        d5 = self.d5(self.d6(q6))
        d3 = self.d3(self.d4(torch.cat((d5, q4), 1)))
        # out = self.d1(self.d2(torch.cat((d3, q2), 1)))
        out = self.d0(self.d1(self.d2(torch.cat((d3, q2), 1))))
        return out


class VectorQuantizerEMA(nn.Module):
    """
         Code taken from: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I
    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, name, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self.name = name

    def forward(self, inputs):
        with record_function(self.name):
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
            input_shape = inputs.shape

            # Flatten input
            flat_input = inputs.view(-1, self._embedding_dim)

            # Calculate distances
            distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(self._embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self._embedding.weight.t())
            )

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self._num_embeddings, device=inputs.device
            )
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self._embedding.weight).view(
                input_shape
            )

            # Use EMA to update the embedding vectors
            if self.training:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                    1 - self._decay
                ) * torch.sum(encodings, 0)

                # Laplace smoothing of the cluster size
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon)
                    * n
                )

                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = nn.Parameter(
                    self._ema_w * self._decay + (1 - self._decay) * dw
                )

                self._embedding.weight = nn.Parameter(
                    self._ema_w / self._ema_cluster_size.unsqueeze(1)
                )

            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self._commitment_cost * e_latent_loss
            # Straight Through Estimator
            quantized = inputs + (quantized - inputs).detach()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            loss,
            quantized.permute(0, 4, 1, 2, 3).contiguous(),
            perplexity,
            encodings,
            encoding_indices,
        )


class FixupBlock(nn.Module):
    """
        Adapted from https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py#L20
    """

    def __init__(self, in_channels, out_channels, identity_function, name):
        super(FixupBlock, self).__init__()

        assert identity_function in ["downsample", "upsample", "level"]

        self.bias1a = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.bias1b = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.bias2a = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.bias2b = nn.Parameter(data=torch.zeros(1), requires_grad=True)

        self.scale = nn.Parameter(data=torch.ones(1), requires_grad=True)

        self.activation = nn.LeakyReLU(inplace=False)

        if identity_function == "downsample":
            self.skip_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

            self.conv1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        elif identity_function == "upsample":
            self.skip_conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )

            self.conv1 = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        elif identity_function == "level":
            self.skip_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

            self.conv1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.nca = nn.Sequential(
            nn.ConstantPad3d(padding=(1, 0, 1, 0, 1, 0), value=0),
            nn.AvgPool3d(kernel_size=2, stride=1),
        )

        self.name = name

    def forward(self, input):
        with record_function(self.name):
            out = self.conv1(input + self.bias1a)
            out = self.nca(self.activation(out + self.bias1b))

            out = self.conv2(out + self.bias2a)
            out = out * self.scale + self.bias2b

            out += self.nca(self.skip_conv(input + self.bias1a))
            out = self.activation(out)

        return out

    def initialize_weights(self, num_layers):

        nn.init.normal_(
            tensor=self.conv1.weight,
            mean=0,
            std=np.sqrt(
                2 / (self.conv1.weight.shape[0] * np.prod(self.conv1.weight.shape[2:]))
            )
            * num_layers ** (-0.5),
        )
        nn.init.constant_(tensor=self.conv2.weight, val=0)
        nn.init.normal_(
            tensor=self.skip_conv.weight,
            mean=0,
            std=np.sqrt(
                2
                / (
                    self.skip_conv.weight.shape[0]
                    * np.prod(self.skip_conv.weight.shape[2:])
                )
            ),
        )


# class SubPixelConvolution3D(nn.Module):
#     def __init__(self, in_channels, out_channels, upsample_factor, name):
#         super(SubPixelConvolution3D, self).__init__()
#         assert upsample_factor == 2
#         self.upsample_factor = upsample_factor
#
#         self.conv = nn.Conv3d(
#             in_channels=in_channels,
#             out_channels=out_channels * 2 ** 3,
#             kernel_size=3,
#             padding=1,
#         )
#
#         self.shuffle = PixelShuffle3D(
#             upscale_factor=upsample_factor, name="PixelShuffle3D"
#         )
#
#         self.nca = nn.Sequential(
#             nn.ConstantPad3d(padding=(1, 0, 1, 0, 1, 0), value=0),
#             nn.AvgPool3d(kernel_size=2, stride=1),
#         )
#
#         self.name = name
#
#     def forward(self, input):
#         with record_function(self.name):
#             out = self.conv(input)
#             out = self.shuffle(out)
#             out = self.nca(out)
#         return out
#
#     def initialize_weights(self):
#         """
#             Code taken from https://github.com/pytorch/pytorch/pull/5429/files
#         """
#         new_shape = [
#             int(self.conv.weight.shape[0] / (self.upsample_factor ** 2))
#         ] + list(self.conv.weight.shape[1:])
#         subkernel = torch.zeros(new_shape)
#         subkernel = nn.init.xavier_normal_(subkernel)
#         subkernel = subkernel.transpose(0, 1)
#
#         subkernel = subkernel.contiguous().view(
#             subkernel.shape[0], subkernel.shape[1], -1
#         )
#
#         kernel = subkernel.repeat(1, 1, self.upsample_factor ** 2)
#
#         transposed_shape = (
#             [self.conv.weight.shape[1]]
#             + [self.conv.weight.shape[0]]
#             + list(self.conv.weight.shape[2:])
#         )
#         kernel = kernel.contiguous().view(transposed_shape)
#
#         kernel = kernel.transpose(0, 1)
#
#         self.conv.weight.data = kernel
#
#
# class PixelShuffle3D(nn.Module):
#     def __init__(self, upscale_factor, name):
#         super(PixelShuffle3D, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.upscale_factor_cubed = upscale_factor ** 3
#         self._shuffle_out = None
#         self._shuffle_in = None
#         self.name = name
#
#     def forward(self, input):
#         with record_function(self.name):
#             shuffle_out = input.new()
#
#             batch_size = input.size(0)
#             channels = int(input.size(1) / self.upscale_factor_cubed)
#             in_depth = input.size(2)
#             in_height = input.size(3)
#             in_width = input.size(4)
#
#             input_view = input.view(
#                 batch_size,  # 0 - 1
#                 channels,  # 1 - 1
#                 self.upscale_factor,  # 2 - 2
#                 self.upscale_factor,  # 3 - 2
#                 self.upscale_factor,  # 4 - 2
#                 in_depth,  # 5 - 5
#                 in_height,  # 6 - 5
#                 in_width,  # 7 - 5
#             )
#
#             shuffle_out.resize_(
#                 input_view.size(0),  # 0 - 1
#                 input_view.size(1),  # 1 - 1
#                 input_view.size(5),  # 2 - 5
#                 input_view.size(2),  # 3 - 2
#                 input_view.size(6),  # 4 - 5
#                 input_view.size(3),  # 5 - 2
#                 input_view.size(7),  # 6 - 5
#                 input_view.size(4),  # 7 - 2
#             )
#
#             shuffle_out.copy_(input_view.permute(0, 1, 5, 2, 6, 3, 7, 4))
#
#             out_depth = in_depth * self.upscale_factor
#             out_height = in_height * self.upscale_factor
#             out_width = in_width * self.upscale_factor
#
#             output = shuffle_out.view(
#                 batch_size, channels, out_depth, out_height, out_width
#             )
#
#         return output
