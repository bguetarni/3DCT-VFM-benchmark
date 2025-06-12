import tqdm, os, glob, argparse
import numpy as np

import artix
import dicom_class

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to ARTIX patients folder')
    parser.add_argument('--output_path', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--task', type=str, required=True, choices=["head_glands_cavities", "headneck_muscles"], help='task of segmentation')
    args = parser.parse_args()

    # input_path = r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data"
    # output_path = r"C:\Users\bilel.guetarni\Desktop\ARTIX\totalsegmentator"

    for p in tqdm.tqdm(glob.glob(os.path.join(args.input_path, "*")), ncols=50):
        print(p)
        patient_data = artix.load_folder(p)
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            # print(ct.path)
            out = ct.apply_totalsegmentator(task=args.task)
            out_path = os.path.join(args.output_path, os.path.split(p)[1], os.path.split(ct.path)[1])
            os.makedirs(out_path, exist_ok=True)
            np.save(os.path.join(out_path, "totalsegmentator_mask.npy"), out)

    print("done")
