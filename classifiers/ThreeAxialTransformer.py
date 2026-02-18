import torch
from torch import nn
from classifiers.classifiers import BaseClassifier
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CenterSpatialCrop, SpatialCrop, Spacing, CastToType
import numpy as np

import dataset_loaders

class Attention(nn.Module):
    def __init__(self, input_dim, seq_length, n_dim, pos_embed, dropout=0.1):
        super().__init__()
        self.n_dim = n_dim

        self.pos_embed = pos_embed
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, seq_length, input_dim))

        self.ln = nn.LayerNorm(input_dim)
        self.to_q = nn.Linear(input_dim, n_dim)
        self.to_k = nn.Linear(input_dim, n_dim)
        self.to_v = nn.Linear(input_dim, n_dim)
        self.proj = nn.Linear(n_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def create_pos_encoding(self, x):
        B, N, D = x.shape
        pe = torch.zeros(B, N, D, device=x.device)
        position = torch.arange(0, N, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=x.device) * (-torch.log(torch.tensor(10000.0)) / D))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        if self.pos_embed:
            pos = self.pos_embed.repeat(x.shape[0], 1, 1)
        else:
            pos = self.create_pos_encoding(x)        
        
        x += pos.to(x.device)
        x_ln = self.ln(x)

        q = self.to_q(x_ln)  # (B, N, n_dim)
        k = self.to_k(x_ln)  # (B, N, n_dim)
        v = self.to_v(x_ln)  # (B, N, n_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.n_dim ** 0.5)  # (B, N, N)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_scores, v)  # (B, N, n_dim)
        h = self.proj(h)
        return self.dropout(h)

class ThreeAxialTransformer(nn.Module, BaseClassifier):
    def __init__(self, dims, n_dim, n_layer, n_class, pos_embed, pool_kernel_size, pool_stride_size, dropout=0.1):
        """
        Args:
            dims (dict): tuple of input dimensions (X,Y,Z)
            n_dim (int): dimension of the axial attention
            n_layer (int): number of axial attention layers
            n_class (int): number of output classes
            pos_embed (bool): whether to use positional embedding
            pool_kernel_size (tuple): pooling kernel size for each dimension
            pool_stride_size (tuple): pooling stride size for each dimension
            dropout (float): dropout rate
        """
        super().__init__()

        if isinstance(pool_kernel_size, int):
            pool_kernel_size = (pool_kernel_size,) * 3

        if isinstance(pool_stride_size, int):
            pool_stride_size = (pool_stride_size,) * 3

        # code preprocess as a dict based on dataset (one specialized for each dataset)
        self.preprocess = {
            "default": Compose([
                LoadImage(ensure_channel_first=True), # Load image and ensure channel dimension
                EnsureType(),                         # Ensure correct data type
                Orientation(axcodes="RAS"),           # Standardize orientation
                Spacing(pixdim=(1.0, 1.0, 1.0)),      # resample to isotropic spacing
                ScaleIntensityRange(                  # Scale intensity to [0,1] range, clipping outliers
                    a_min=-1024,
                    a_max=2048,
                    b_min=0,
                    b_max=1,
                    clip=True
                ),
                # the next three transforms are to ensure the input contains the H&N region
                SpatialCrop(roi_start=(0,0,0), roi_end=(512,512,300)),                
                CenterSpatialCrop(roi_size=dims),
                CastToType(torch.float32),
            ])}

        for name_, transforms_ in [("artix", dataset_loaders.ARTIX), 
                                   ("hecktor", dataset_loaders.HECKTOR), 
                                   ("headneckctatlas", dataset_loaders.HeadNeckCTAtlas),
                                   ("headneckpetct", dataset_loaders.HeadNeckPETCT), 
                                   ("oropharyngealradiomicsoutcomes", dataset_loaders.OropharyngealRadiomicsOutcomes),
                                   ("qinheadneck", dataset_loaders.QINHEADNECK), 
                                   ("radcure", dataset_loaders.RADCURE)]:
            self.preprocess[name_] = Compose([
                LoadImage(ensure_channel_first=True), # Load image and ensure channel dimension
                EnsureType(),                         # Ensure correct data type
                Orientation(axcodes="RAS"),           # Standardize orientation
                Spacing(pixdim=(1.0, 1.0, 1.0)),      # resample to isotropic spacing
                ScaleIntensityRange(                  # Scale intensity to [0,1] range, clipping outliers
                    a_min=-1024,
                    a_max=2048,
                    b_min=0,
                    b_max=1,
                    clip=True
                ),
                # the next three transforms are to ensure the input contains the H&N region
                *transforms_.get_spatial_transforms(),
                CenterSpatialCrop(roi_size=dims),
                CastToType(torch.float32),
            ])

        # three axial attention layers
        self.layers = nn.ModuleList()
        dim_ = dims
        for _ in range(n_layer):
            axial_layers = nn.ModuleList()
            for i, d in enumerate(dim_):
                seq_length = np.prod([k for j,k in enumerate(dim_) if j != i]) # sequence length along dimension d
                axial_layers.append(Attention(input_dim=d, seq_length=seq_length, n_dim=n_dim, pos_embed=pos_embed, dropout=dropout))
            axial_layers.append(nn.AvgPool3d(kernel_size=pool_kernel_size, stride=pool_stride_size, ceil_mode=False))
            self.layers.append(axial_layers)
            dim_ = [int(np.floor((dim_[i] - pool_kernel_size[i]) / pool_stride_size[i] + 1)) for i in range(len(dim_))]

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(np.prod(dim_), n_class)

    def vol2seq(self, x, dim):
        x = x.movedim(dim, -1)  # move the specified dim to be sequence dim
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # flatten the other dims
        return x

    def seq2vol(self, x, dim, original_shape):
        b = x.shape[0]
        new_shape = tuple(d for i,d in enumerate(original_shape) if i+1 != dim) # exclude the specified dim
        x = x.reshape(b, *new_shape, x.shape[-1])  # reshape to volume shape without the specified dim
        x = x.movedim(-1, dim)  # move back to original position
        return x

    def forward(self, x):
        """
        Args
            x (torch.Tensor): Input tensor of shape (B, W, H, D)
        Returns
            out (torch.Tensor): Output logits of shape (B, num_classes)
        """
        for layer in self.layers:
            for dim_, ax in enumerate(layer[:-1]):
                original_shape = x.shape[1:]  # save original shape
                x = self.vol2seq(x, dim_+1) # convert volume to sequence along axis d
                x = ax(x)  # axial attention
                x = self.seq2vol(x, dim_+1, original_shape) # convert back to volume
            x = layer[-1](x)  # pooling
        h = torch.flatten(x, start_dim=1, end_dim=-1)
        h = self.dropout(h)
        out = self.head(h)
        return out
    
    def preprocess_(self, input):
        if input["dataset"] in self.preprocess.keys():
            return self.preprocess[input["dataset"]](input["image"])
        else:
            return self.preprocess["default"](input["image"])

    def __call__(self, input, device="cpu"):
        input_tensor = torch.cat(list(map(self.preprocess_, input)), dim=0).to(device)
        output = self.forward(input_tensor)
        return output
    
    def load(self, path_):
        return self.load_state_dict(torch.load(path_))
    
    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
