import torch


class VectorQuantizerEMA(torch.nn.Module):
    # Code taken from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, name, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = torch.nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self.name = name

    def forward(self, inputs):
        with torch.autograd.profiler.record_function(self.name):
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
            input_shape = inputs.shape

            flat_input = inputs.view(-1, self._embedding_dim)

            distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(self._embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self._embedding.weight.t())
            )

            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self._num_embeddings, device=inputs.device
            )
            encodings.scatter_(1, encoding_indices, 1)

            quantized = torch.matmul(encodings, self._embedding.weight).view(
                input_shape
            )

            if self.training:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                    1 - self._decay
                ) * torch.sum(encodings, 0)

                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon)
                    * n
                )

                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = torch.nn.Parameter(
                    self._ema_w * self._decay + (1 - self._decay) * dw
                )

                self._embedding.weight = torch.nn.Parameter(
                    self._ema_w / self._ema_cluster_size.unsqueeze(1)
                )

            e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
            loss = self._commitment_cost * e_latent_loss

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
