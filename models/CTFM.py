import torch
from lighter_zoo import SegResEncoder
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CenterSpatialCrop, Flip, SpatialCrop

def infer(input, preprocess, model, device):
    with torch.no_grad():
        input_tensor = preprocess(input)
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
        LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
        EnsureType(),                         # Ensure correct data type
        Orientation(axcodes="SPL"),           # Standardize orientation
        # Scale intensity to [0,1] range, clipping outliers
        ScaleIntensityRange(
            a_min=-1024,    # Min HU value
            a_max=2048,     # Max HU value
            b_min=0,        # Target min
            b_max=1,        # Target max
            clip=True       # Clip values outside range
        ),

        Flip(spatial_axis=-1),
        SpatialCrop(roi_start=(0,0,0), roi_end=(200,512,512)),
        Flip(spatial_axis=-1),
        CenterSpatialCrop(roi_size=(200,350,350)),
    ])

    return infer, preprocess, model
