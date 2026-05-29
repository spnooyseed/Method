"""
Train a diffusion model for recommendation
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(
        self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5
    ):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert (
            out_dims[0] == in_dims[-1]
        ), "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])
            ]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        mean_type,
        noise_schedule,
        noise_scale,
        noise_min,
        noise_max,
        steps,
        device,
        history_num_per_term=10,
        beta_fixed=True,
    ):

        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(
            device
        )
        self.Lt_count = th.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.0:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = (
                    0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                )
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (
                len(self.betas) == self.steps
            ), "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (
                self.betas <= 1
            ).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(
                    self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
                )
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif (
            self.noise_schedule == "binomial"
        ):  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat(
            [th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        ).to(
            self.device
        )  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat(
            [self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]
        ).to(
            self.device
        )  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.0:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = (
                    out["mean"]
                    + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                )
            else:
                x_t = out["mean"]
        return x_t

    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, "importance")
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output))
                    ** 2
                    / 2.0
                )
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss

        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def sample_timesteps(
        self, batch_size, device, method="uniform", uniform_prob=0.001
    ):
        if method == "importance":  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")

            Lt_sqrt = th.sqrt(th.mean(self.Lt_history**2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1.0 < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == "uniform":  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + self._extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


import numpy as np
import random
import torch
import scipy.sparse as sp
import os
from torch.utils.data import Dataset
import pickle


def data_load(train_path, test_path):
    with open(train_path, "rb") as fs:
        trnMat = (pickle.load(fs) != 0).astype(np.float32)
    if type(trnMat) is not sp.coo_matrix:
        trnMat = sp.coo_matrix(trnMat)
    with open(test_path, "rb") as fs:
        test_list = (pickle.load(fs) != 0).astype(np.float32)
    if type(test_list) is not sp.coo_matrix:
        test_list = sp.coo_matrix(test_list)

    n_user, n_item = trnMat.shape
    print(f"user num: {n_user}")
    print(f"item num: {n_item}")

    return trnMat.tocsr(), test_list.tocsr(), n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)


import numpy as np
import torch
import math


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = 1.0 / (j + 1.0)
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if idcg != 0:
                    ndcg += dcg / idcg

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

    return precision, recall, NDCG, MRR


def print_results(loss, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    # if valid_result is not None:
    #     print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
    #                         '-'.join([str(x) for x in valid_result[0]]),
    #                         '-'.join([str(x) for x in valid_result[1]]),
    #                         '-'.join([str(x) for x in valid_result[2]]),
    #                         '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print(
            "[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                "-".join([str(x) for x in test_result[0]]),
                "-".join([str(x) for x in test_result[1]]),
                "-".join([str(x) for x in test_result[2]]),
                "-".join([str(x) for x in test_result[3]]),
            )
        )


"""
Train a diffusion model for recommendation
"""

import argparse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


import random

random_seed = 1
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="yelp_clean", help="choose the dataset"
)
parser.add_argument(
    "--data_path", type=str, default="./Datasets/", help="load data path"
)
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=100, help="upper epoch limit")
parser.add_argument("--topN", type=str, default="[10, 20]")
parser.add_argument("--tst_w_val", action="store_true", help="test with validation")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument(
    "--save_path", type=str, default="./saved_models/", help="save model path"
)
parser.add_argument("--log_name", type=str, default="log", help="the log name")
parser.add_argument("--round", type=int, default=1, help="record the experiment")

# params for the model
parser.add_argument("--time_type", type=str, default="cat", help="cat or add")
parser.add_argument("--dims", type=str, default="[1000]", help="the dims for the DNN")
parser.add_argument(
    "--norm", type=bool, default=False, help="Normalize the input or not"
)
parser.add_argument("--emb_size", type=int, default=10, help="timestep embedding size")

# params for diffusion
parser.add_argument(
    "--mean_type", type=str, default="x0", help="MeanType for diffusion: x0, eps"
)
parser.add_argument("--steps", type=int, default=50, help="diffusion steps")
parser.add_argument(
    "--noise_schedule",
    type=str,
    default="linear-var",
    help="the schedule for noise generating",
)
parser.add_argument(
    "--noise_scale",
    type=float,
    default=0.00001,
    help="noise scale for noise generating",
)
parser.add_argument(
    "--noise_min",
    type=float,
    default=0.0001,
    help="noise lower bound for noise generating",
)
parser.add_argument(
    "--noise_max",
    type=float,
    default=0.02,
    help="noise upper bound for noise generating",
)
parser.add_argument(
    "--sampling_noise", type=bool, default=False, help="sampling with noise or not"
)
parser.add_argument(
    "--sampling_steps",
    type=int,
    default=0,
    help="steps of the forward process during inference",
)
parser.add_argument(
    "--reweight",
    type=bool,
    default=True,
    help="assign different weight to different timestep or not",
)

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print(
    "Starting time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
)

### DATA LOAD ###
train_path = os.path.join(args.data_path, args.dataset, "trnMat.pkl")
test_path = os.path.join(args.data_path, args.dataset, "tstMat.pkl")

train_data, test_y_data, n_user, n_item = data_load(train_path, test_path)
train_dataset = DataDiffusion(torch.FloatTensor(train_data.toarray()))
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,
)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = DataDiffusion(
        torch.FloatTensor(train_data.toarray())
    )  # + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data  # + valid_y_data

print("data ready.")


### Build Gaussian Diffusion ###
if args.mean_type == "x0":
    mean_type = ModelMeanType.START_X
elif args.mean_type == "eps":
    mean_type = ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)
diffusion = GaussianDiffusion(
    mean_type,
    args.noise_schedule,
    args.noise_scale,
    args.noise_min,
    args.noise_max,
    args.steps,
    device,
).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(
    device
)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)


def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[
                e_idxlist[
                    batch_idx * args.batch_size : batch_idx * args.batch_size
                    + len(batch)
                ]
            ]
            batch = batch.to(device)
            prediction = diffusion.p_sample(
                model, batch, args.sampling_steps, args.sampling_noise
            )
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results


best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print("-" * 18)
        print("Exiting from training early")
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        if args.tst_w_val:
            test_results = evaluate(
                test_twv_loader, test_y_data, mask_tv, eval(args.topN)
            )
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        print_results(None, test_results)

        if test_results[1][1] > best_recall:  # recall@20 as selection
            best_recall, best_epoch = test_results[1][1], epoch
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(
                model,
                "{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth".format(
                    args.save_path,
                    args.dataset,
                    args.lr,
                    args.weight_decay,
                    args.batch_size,
                    args.dims,
                    args.emb_size,
                    args.mean_type,
                    args.steps,
                    args.noise_scale,
                    args.noise_min,
                    args.noise_max,
                    args.sampling_steps,
                    args.reweight,
                    args.log_name,
                ),
            )

    print(
        "Runing Epoch {:03d} ".format(epoch)
        + "train loss {:.4f}".format(total_loss)
        + " costs "
        + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    )
    print("---" * 18)

print("===" * 18)
print("End. Best Epoch {:03d} ".format(best_epoch))
print_results(None, best_test_results)
print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
