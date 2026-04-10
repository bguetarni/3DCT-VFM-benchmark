import os
import argparse
import pickle
import logging
import datetime
from dataloader import cohorts_map
from dataloader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action="store_true", default=False, help='weither to overwrite dataset if already existing')
    parser.add_argument('--input', type=str, required=True, help='path to dataset patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--cohort', type=str, required=True, choices=list(cohorts_map.keys()), 
                        help='which cohort to build (change for certain parts)')
    args = parser.parse_args()

    output = os.path.join(args.output, f"{args.cohort}.pickle")
    if os.path.exists(output) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({args.output}). To overwrite, set argument --overwrite to True")
        exit(0)

    loader = cohorts_map[args.cohort](args.input)
    
    logging.basicConfig(
                filename=f"./{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
                format='%(levelname)s: %(message)s',
                filemode='w',
                level=logging.DEBUG)
    logger = logging.getLogger()

    data = loader.build(logger)
    print(f"\n {len(data)} patients loaded")

    print("saving in pkl..")
    with open(output, "wb") as f:
        pickle.dump(data, f)

    print("Done")
