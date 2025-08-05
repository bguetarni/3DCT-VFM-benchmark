import glob, os, tqdm, argparse, pickle
import artix
import hnscc3dctrt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to dataset patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--cohort', type=str, required=True, choices=["artix", "tcia"], help='which cohort to build (change for certain parts)')
    parser.add_argument('--id_map', type=str, default=None, help='path to xlsx file with ID mapping correlation for ARTIX')
    parser.add_argument('--clinical', type=str, required=True, help='path to clinical file or folder')
    parser.add_argument('--save_freq', type=int, default=1, help='frequency (no of patients) to save')
    args = parser.parse_args()

    data = []
    all_patients_to_load = glob.glob(os.path.join(args.input, "*"))
    for i, p in tqdm.tqdm(enumerate(all_patients_to_load), total=len(all_patients_to_load), ncols=50):
        if not(os.path.isdir(p)):
            continue

        if args.cohort == "artix":
            p = artix.load_patient(path=p, id_map=args.id_map, clinical=args.clinical, log="./log")
        elif args.cohort == "tcia":
            p = hnscc3dctrt.load_patient(path=p, clinical=args.clinical, log="./log")
        else:
            continue
        
        # set base path
        p.update_study_base_path(args.input)

        data.append(p)
        
        if ((i+1) % args.save_freq) == 0:
            print("saving in pkl..")
            with open(args.output, "wb") as f:
                pickle.dump(data, f)

    print(f"\n {len(data)} patients loaded")

    print("saving in pkl..")
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print("Done")
