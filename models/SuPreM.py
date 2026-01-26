import os
import torch
from monai.networks.nets import SegResNet
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, ToTensord, CenterSpatialCropd, Flipd, SpatialCropd, AddChanneld, Orientationd, Spacingd
from utils import BboxCropd

def infer(input, bbox, preprocess, model, device):
    with torch.no_grad():
        input_tensor = preprocess({"image": input, "bbox": bbox})
        input_tensor = input_tensor["image"]
        input_tensor = input_tensor.unsqueeze(dim=0).to(device)
        output = model.encode(input_tensor)[0].cpu()
        avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1)
        return avg_output.numpy()

def load(device, checkpoint=None):
    if not(checkpoint):
        checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), "supervised_suprem_segresnet_2100.pth")
    
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            BboxCropd(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),            
            ToTensord(keys=["image"]),
        ])

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=25,
        dropout_prob=0.0,
    )

    store_dict = model.state_dict()
    model_dict = torch.load(checkpoint, map_location='cpu')['net']
    new_model_dict={}
    for key, value in model_dict.items():
        new_key = key.replace('module.', '')
        new_model_dict[key] = value
    model_dict = new_model_dict  
    amount = 0
    for key in model_dict.keys():
        new_key = '.'.join(key.split('.')[1:])
        if new_key in store_dict.keys():
            store_dict[new_key] = model_dict[key]   
            amount += 1
    assert amount == len(store_dict), "the model is not loaded successfully"
    model.eval().to(device=device)
    return infer, test_transforms, model
