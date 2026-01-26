import torch
from lighter_zoo import SegResEncoder
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, ScaleIntensityRanged
from . import utils

def infer(input, bbox, preprocess, model, device):
    with torch.no_grad():
        input_tensor = preprocess({"image": input, "bbox": bbox})
        input_tensor = input_tensor["image"]
        input_tensor = input_tensor.unsqueeze(dim=0).to(device)
        output = model(input_tensor)[-1].cpu()
        avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1)
        return avg_output.numpy()


def load(device):
    # Load pre-trained model
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model.eval().to(device=device)

    # Preprocessing pipeline
    preprocess = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        utils.BboxCropd(keys=["image"]),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="SPL"),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,    # Min HU value
            a_max=2048,     # Max HU value
            b_min=0,        # Target min
            b_max=1,        # Target max
            clip=True       # Clip values outside range
        ),
    ])

    return infer, preprocess, model
