import os, argparse, pickle
from dataloader import ARTIX, TCIA_HNSCC3DCTRT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action="store_true", default=False, help='weither to overwrite dataset if already existing')
    parser.add_argument('--input', type=str, required=True, help='path to dataset patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--cohort', type=str, required=True, choices=["artix", "tcia"], help='which cohort to build (change for certain parts)')
    parser.add_argument('--id_map', type=str, default=None, help='path to xlsx file with ID mapping correlation for ARTIX')
    parser.add_argument('--clinical', type=str, required=True, help='path to clinical file or folder')
    args = parser.parse_args()

    if os.path.exists(args.output) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({args.output}). To overwrite, set argument --overwrite to True")
        exit(0)

    if args.cohort == "artix":
        loader = ARTIX(args.input, args.id_map)
    elif args.cohort == "tcia":
        loader = TCIA_HNSCC3DCTRT(args.input)
    else:
        raise ValueError(f"argument --cohort value not implemented: {args.cohort}")

    data = loader.load(args.clinical)
    print(f"\n {len(data)} patients loaded")

    print("saving in pkl..")
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print("Done")
