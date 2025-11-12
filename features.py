import tqdm, os, argparse, pickle, time
import pandas
import torch

from models import CTFM, ModelGenesis, SuPreM, LLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save features')
    parser.add_argument('--tmp_folder', type=str, required=True, help='path where temporary files (Nifti, DICOM) are saved')
    parser.add_argument('--overwrite', action="store_true", default=False, help='weither to overwrite features file if already existing')
    parser.add_argument('--type', type=str, default=False, choices=["ct-fm", "suprem", "model-genesis", "llm"], help="type of features")
    parser.add_argument('--cohort', type=str, required=True, choices=["artix", "hecktor", "headneckctatlas", "headneckpetct", "hnscc3dctrt", "oropharyngealradiomicsoutcomes", "qinheadneck", "radcure"], 
                        help='which cohort to build (change for certain parts)')
    parser.add_argument('--description', type=str, help='path where Microsoft Copilot descriptions are saved')
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    if args.type == "llm" and args.description is None:
        raise ValueError("If argument type is llm, then argument description must be provided")

    out_path = os.path.join(args.output, args.cohort, f"{args.type}.csv")
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    if os.path.exists(out_path) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({out_path}). To overwrite, set argument --overwrite to True")
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device ", device)

    with open(args.input, "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)
        print(f"found {len(patients)} patients in {args.input}")

    # create temporary folder for saving nifti files
    os.makedirs(args.tmp_folder, exist_ok=True)
    ct_nii_path = os.path.join(args.tmp_folder, f"volume.{int(time.time()*10)}.nii.gz")

    features = []
    for id_, p in tqdm.tqdm(list(patients.items())[:2], ncols=50):   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if args.type != "llm":
            try:
                p.sort_imaging()   # sort images to recover first CT scans (i.e., CT0)
                p.ct[0].convert2nifti(ct_nii_path)   # convert first scan
            except (ValueError, KeyError):
                continue
        else:
            df = pandas.read_csv(os.path.join(args.description, f"{args.cohort}.csv"))
            id_col, desc_col = df.columns
            
            if args.cohort == "artix":   # handle specific case
                df[id_col] = df[id_col].apply(lambda i: str(i).zfill(3))
            
            ct_nii_path = df[df[id_col] == id_][desc_col].item()

        match args.type:
            case "ct-fm":
                infer = CTFM.load(device)
            case "suprem":
                infer = SuPreM.load(device)
            case "model-genesis":
                infer = ModelGenesis.load(device)
            case "llm":
                infer = LLM.load(device)
            case _:
                raise ValueError(f"argument --type value not implemented: {args.type}")
        
        output = infer(ct_nii_path)
        fts = output.flatten()
        for j, f in enumerate(fts):
            features.append({"name": j, "value": f, "features": args.type, "patient": id_})

    # save to csv
    pandas.DataFrame(features).to_csv(out_path, index=False)
    print("Done.")
