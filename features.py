import tqdm
import os
import argparse
import pickle
import pandas
import torch

from models import CTFM, SuPreM, VISTA3D
from dataloader import cohorts_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to folder containing cohorts PICKLE files')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save features')
    parser.add_argument('--overwrite', action="store_true", default=False, help='whether to overwrite features file if already existing')
    parser.add_argument('--type', type=str, required=True, choices=["ct-fm", "suprem", "vista3d"], help="type of features")
    parser.add_argument('--cohort', type=str, required=True, choices=list(cohorts_map.keys()), 
                        help='which cohort to build (change for certain parts)')
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    out_path = os.path.join(args.output, args.cohort, f"{args.type}.csv")
    
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    if os.path.exists(out_path) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({out_path}). To overwrite, set argument --overwrite to True")
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device ", device)

    with open(os.path.join(args.input, f"{args.cohort}.pickle"), "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)
        print(f"found {len(patients)} patients in {args.input}")

    print(f"loading model {args.type}...")
    match args.type:
        case "ct-fm":
            infer, preprocess, model = CTFM.load(device)
        case "suprem":
            infer, preprocess, model = SuPreM.load(device)
        case "vista3d":
            infer, preprocess, model = VISTA3D.load(device)
        case _:
            raise ValueError(f"argument --type value not implemented: {args.type}")

    features = []
    for id_, p in tqdm.tqdm(list(patients.items()), ncols=50):
        try:
            p.sort_imaging()   # sort images to recover first CT scan (i.e., CT0)
            input_path = p.ct[0].path
            bbox = p.ct[0].get_GTV_bbox()
        except (ValueError, KeyError):
            continue
        
        output = infer(input_path, bbox, preprocess, model, device)
        fts = output.flatten()
        for j, f in enumerate(fts):
            features.append({"name": str(j), "value": f, "features": args.type, "patient": id_})

    # save to csv
    pandas.DataFrame(features).to_csv(out_path, index=False)
    print("Done.")
