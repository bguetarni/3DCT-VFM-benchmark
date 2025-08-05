import tqdm, os, glob, argparse, pathlib, pickle
import numpy as np

import artix
import dicom_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output_path', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--task', type=str, required=True, choices=["head_glands_cavities", "headneck_muscles", "craniofacial_structures"], help='task of segmentation')
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.input_path, "rb") as f:
        patients = pickle.load(f)

    print(f"found {len(patients)} patients in {args.input_path}")

    for p in tqdm.tqdm(patients, ncols=50):
        for ct in p.ct:
            out = ct.apply_totalsegmentator(task=args.task)
            out_path = os.path.join(args.output_path, pathlib.Path(p).name, pathlib.Path(ct.path).name)
            os.makedirs(out_path, exist_ok=True)
            np.save(os.path.join(out_path, f"{args.task}.npy"), out)

    print("done")
