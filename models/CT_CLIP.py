import os
from collections import OrderedDict
import torch
from torch.nn import functional as F
from monai.transforms import MapTransform, Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, ToTensord, \
    ScaleIntensityRanged, Spacingd, Orientationd, SpatialPadd, CenterSpatialCropd

from transformer_maskgit import CTViT
from . import utils

class CropZd(MapTransform):
    """
    Make sure z dimension is divisible by z_div_factor
    """
    def __init__(self, keys, z_div_factor=10, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.z_div_factor = z_div_factor

    def __call__(self, data):
        if self.z_div_factor:
            z = data["image"].shape[-1]
            z_max = z - (z % self.z_div_factor)
            data["image"] = data["image"][..., :z_max]
        return data


def infer(input, bbox, preprocess, model, device):
    with torch.no_grad():
        input_tensor = preprocess({"image": input, "bbox": bbox})
        input_tensor = input_tensor["image"]
        input_tensor = input_tensor.unsqueeze(dim=0).to(device)
        output = model(input_tensor, return_encoded_tokens=True).cpu()
        avg_output = F.adaptive_avg_pool3d(torch.transpose(output, 1, -1), (1,1,1))
        return avg_output.numpy()


def load(device, checkpoint=None):
    if not(checkpoint):
        checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CT-CLIP_v2.pt")
    
    # create model
    model = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10, 
                  spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)

    # load weights
    pattrn = "visual_transformer."
    d = OrderedDict({k.removeprefix(pattrn): v for k, v in torch.load(checkpoint).items() if k.startswith(pattrn)})
    model.load_state_dict(d)
    model.eval().to(device=device)

    # Preprocessing pipeline
    preprocess = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        utils.BboxCropd(keys=["image"], roi_size=(480,480,10)),   # crop to GTV bounding box
        Spacingd(keys=["image"], pixdim=(0.75, 0.75, 1.5), mode=("bilinear")),   # resample to common spacing
        CropZd(keys=["image"], z_div_factor=10),   # make sure z dimension is divisible by 10 after resampling (requirement of the model)
        CenterSpatialCropd(keys=["image"], roi_size=(480,480,-1)),   # crop to common size (in case bbox crop results in larger size than expected, we center-crop to expected size)
        SpatialPadd(keys=["image"], spatial_size=(480,480,-1)),   # pad to common size (in case bbox crop results in smaller size than expected)
        Orientationd(keys=["image"], axcodes="SLP"),   # reorient to common orientation
        EnsureTyped(keys=["image"]),        
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=-1,
            b_max=1,
            clip=True,
        ),
        ToTensord(keys=["image"]),
    ])

    return infer, preprocess, model
