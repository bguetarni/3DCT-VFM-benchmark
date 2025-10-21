import os, argparse, pickle, logging
from datetime import datetime
from dataloader import ARTIX, HECKTOR, HeadNeckCTAtlas, HeadNeckPETCT, HNSCC3DCTRT, OropharyngealRadiomicsOutcomes, QINHEADNECK, RADCURE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action="store_true", default=False, help='weither to overwrite dataset if already existing')
    parser.add_argument('--input', type=str, required=True, help='path to dataset patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--cohort', type=str, required=True, choices=["artix", "hecktor", "headneckctatlas", "headneckpetct", "hnscc3dctrt", "oropharyngealradiomicsoutcomes", "qinheadneck", "radcure"], 
                        help='which cohort to build (change for certain parts)')
    args = parser.parse_args()

    output = os.path.join(args.output, f"{args.cohort}.pickle")
    if os.path.exists(output) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({args.output}). To overwrite, set argument --overwrite to True")
        exit(0)

    match args.cohort:
        case "artix":
            loader = ARTIX(args.input)
        case "hecktor":
            loader = HECKTOR(args.input)
        case "headneckctatlas":
            loader = HeadNeckCTAtlas(args.input)
        case "headneckpetct":
            loader = HeadNeckPETCT(args.input)
        case "hnscc3dctrt":
            loader = HNSCC3DCTRT(args.input)
        case "oropharyngealradiomicsoutcomes":
            loader = OropharyngealRadiomicsOutcomes(args.input)
        case "qinheadneck":
            loader = QINHEADNECK(args.input)
        case "radcure":
            loader = RADCURE(args.input)
        case _:
            raise ValueError(f"argument --cohort value not implemented: {args.cohort}")
    
    logging.basicConfig(
                filename=f"./{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
                format='%(levelname)s: %(message)s',
                filemode='w',
                level=logging.DEBUG)
    logger = logging.getLogger()

    data = loader.load(logger)
    print(f"\n {len(data)} patients loaded")

    print("saving in pkl..")
    with open(output, "wb") as f:
        pickle.dump(data, f)

    print("Done")
