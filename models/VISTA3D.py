# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode, has_option, ensure_tuple

from collections import OrderedDict
from monai.networks.layers.factories import Norm, split_args, Act
from monai.bundle import ConfigParser
from . import utils

# __all__ = ["SegResNetDS2"]


def scales_for_resolution(resolution: tuple | list, n_stages: int | None = None):
    """
    A helper function to compute a schedule of scale at different downsampling levels,
    given the input resolution.

    .. code-block:: python

        scales_for_resolution(resolution=[1,1,5], n_stages=5)

    Args:
        resolution: input image resolution (in mm)
        n_stages: optionally the number of stages of the network
    """

    ndim = len(resolution)
    res = np.array(resolution)
    if not all(res > 0):
        raise ValueError("Resolution must be positive")

    nl = np.floor(np.log2(np.max(res) / res)).astype(np.int32)
    scales = [tuple(np.where(2**i >= 2**nl, 1, 2)) for i in range(max(nl))]
    if n_stages and n_stages > max(nl):
        scales = scales + [(2,) * ndim] * (n_stages - max(nl))
    else:
        scales = scales[:n_stages]
    return scales


def aniso_kernel(scale: tuple | list):
    """
    A helper function to compute kernel_size, padding and stride for the given scale

    Args:
        scale: scale from a current scale level
    """
    kernel_size = [3 if scale[k] > 1 else 1 for k in range(len(scale))]
    padding = [k // 2 for k in kernel_size]
    return kernel_size, padding, scale


class SegResBlock(nn.Module):
    """
    Residual network block used SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: tuple | int = 3,
        act: tuple | str = "relu",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """
        super().__init__()

        if isinstance(kernel_size, (tuple, list)):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2  # type: ignore

        self.norm1 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        self.act1 = get_act_layer(act)
        self.conv1 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm2 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        self.act2 = get_act_layer(act)
        self.conv2 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        x += identity
        return x


class SegResEncoder(nn.Module):
    """
    SegResEncoder based on the econder structure in `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        head_module: optional callable module to apply to the final features.
        anisotropic_scales: optional list of scale for each scale level.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_down: tuple = (1, 2, 2, 4),
        head_module: nn.Module | None = None,
        anisotropic_scales: tuple | None = None,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("`spatial_dims` can only be 1, 2 or 3.")

        # ensure normalization has affine trainable parameters (if not specified)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)  # type: ignore

        # ensure activation is inplace (if not specified)
        act = split_args(act)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)  # type: ignore

        filters = init_filters  # base number of features

        kernel_size, padding, _ = (
            aniso_kernel(anisotropic_scales[0]) if anisotropic_scales else (3, 1, 1)
        )
        self.conv_init = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=False,
        )
        self.layers = nn.ModuleList()

        for i in range(len(blocks_down)):
            level = nn.ModuleDict()

            kernel_size, padding, stride = (
                aniso_kernel(anisotropic_scales[i]) if anisotropic_scales else (3, 1, 2)
            )
            blocks = [
                SegResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=filters,
                    kernel_size=kernel_size,
                    norm=norm,
                    act=act,
                )
                for _ in range(blocks_down[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)

            if i < len(blocks_down) - 1:
                level["downsample"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters,
                    out_channels=2 * filters,
                    bias=False,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            else:
                level["downsample"] = nn.Identity()

            self.layers.append(level)
            filters *= 2

        self.head_module = head_module
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.init_filters = init_filters
        self.norm = norm
        self.act = act
        self.spatial_dims = spatial_dims

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        x = self.conv_init(x)

        for level in self.layers:
            x = level["blocks"](x)
            outputs.append(x)
            x = level["downsample"](x)

        if self.head_module is not None:
            outputs = self.head_module(outputs)

        return outputs[-1]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._forward(x)


def infer(input, bbox, preprocess, model, device):
    with torch.no_grad():
        input_tensor = preprocess({"image": input, "bbox": bbox})
        input_tensor = input_tensor["image"]
        input_tensor = input_tensor.unsqueeze(dim=0).to(device)
        output = model(input_tensor).cpu()
        avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1)
        return avg_output.numpy()

def load(device, checkpoint=None, infer_yaml=None):
    if not(checkpoint):
        checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pt")

    if not(infer_yaml):
        infer_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), "infer.yaml")

    spatial_dims = 3
    init_filters = 48
    in_channels = 1
    out_channels = 48
    act = "relu"
    norm = "instance"
    blocks_down = (1, 2, 2, 4, 4)
    blocks_up = None
    dsdepth = 1
    preprocess = None
    upsample_mode = "deconv"
    resolution = None

    if resolution is not None:
        if not isinstance(resolution, (list, tuple)):
            raise TypeError("resolution must be a tuple")
        elif not all(r > 0 for r in resolution):
            raise ValueError("resolution must be positive")

    # ensure normalization had affine trainable parameters (if not specified)
    norm = split_args(norm)
    if has_option(Norm[norm[0], spatial_dims], "affine"):
        norm[1].setdefault("affine", True)  # type: ignore

    # ensure activation is inplace (if not specified)
    act = split_args(act)
    if has_option(Act[act[0]], "inplace"):
        act[1].setdefault("inplace", True)  # type: ignore

    anisotropic_scales = None
    if resolution:
        anisotropic_scales = scales_for_resolution(
            resolution, n_stages=len(blocks_down)
        )
    anisotropic_scales = anisotropic_scales

    model = SegResEncoder(
        spatial_dims=spatial_dims,
        init_filters=init_filters,
        in_channels=in_channels,
        act=act,
        norm=norm,
        blocks_down=blocks_down,
        anisotropic_scales=anisotropic_scales,
        )

    pattrn = "image_encoder.encoder."
    d = OrderedDict({k.removeprefix(pattrn): v for k, v in torch.load("models/model.pt").items() if k.startswith(pattrn)})
    model.load_state_dict(d)
    model.eval().to(device)

    parser = ConfigParser()
    parser.read_config(infer_yaml)
    infer_transforms = parser.get_parsed_content("transforms_infer")

    # insert BboxCropd after LoadImaged (which is 0th)
    tr_ = list(infer_transforms.transforms)
    tr_.insert(1, utils.BboxCropd(keys=["image"]))
    infer_transforms.transforms = ensure_tuple(tr_)
    
    return infer, infer_transforms, model
