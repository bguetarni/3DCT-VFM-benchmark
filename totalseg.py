import tqdm, os, argparse, pathlib, pickle, re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output_path', type=str, required=True, help='path to folder to save results')
    parser.add_argument('--nii_path', type=str, required=True, help='path where nifti data is or where to save it')
    parser.add_argument('--task', type=str, required=True, choices=["head_glands_cavities", "headneck_muscles", "craniofacial_structures"], help='task of segmentation')
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.input_path, "rb") as f:
        patients = pickle.load(f)

    print(f"found {len(patients)} patients in {args.input_path}")

    for p in tqdm.tqdm(patients, ncols=50):
        for ct in p.ct:
            nii_path = os.path.join(args.nii_path, str(p.id), f"{pathlib.Path(ct.path).name}.nii.gz")
            nii_path = re.sub('__+', '_', nii_path) # dcm2niix naming compatibility
            out_path = os.path.join(args.output_path, str(p.id), pathlib.Path(ct.path).name, f"{args.task}.nii.gz")
            out_path = re.sub('__+', '_', out_path) # dcm2niix naming compatibility

            os.makedirs(os.path.split(nii_path)[0], exist_ok=True)
            os.makedirs(os.path.split(out_path)[0], exist_ok=True)

            ct.apply_totalsegmentator(task=args.task, tmp_nii_input=nii_path, tmp_nii_output=out_path)

    print("done")
