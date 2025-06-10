import glob, os, tqdm, argparse, pickle
import artix

# id_map=r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\ARTIX_ID_CORRELATION.xlsx",
# clinical_csv=[
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_PATIENT_DESCRIPTION_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_EFFICACY_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_SALIVATION_DATA_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_TREATMENT_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_DOSIMETRY_LTSI.csv",

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to ARTIX patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--id_map', type=str, required=True, help='path to xlsx file with ID mapping correlation')

    # clinical csv files
    parser.add_argument('--PATIENT_DESCRIPTION', type=str, required=True, help='path to PATIENT_DESCRIPTION csv')
    parser.add_argument('--EFFICACY', type=str, required=True, help='path to EFFICACY csv')
    parser.add_argument('--SALIVATION_DATA', type=str, required=True, help='path to SALIVATION_DATA csv')
    parser.add_argument('--TREATMENT', type=str, required=True, help='path to TREATMENT csv')
    parser.add_argument('--DOSIMETRY', type=str, required=True, help='path to DOSIMETRY csv')
    
    args = parser.parse_args()

    path = r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data"
    patients = []
    for p in tqdm.tqdm(glob.glob(os.path.join(path, "*")), ncols=50):
        patients.append(artix.load_patient(
            path=p,
            id_map=args.id_map,
            clinical_csv=[args.PATIENT_DESCRIPTION, args.EFFICACY, args.SALIVATION_DATA, args.TREATMENT, args.DOSIMETRY],
            log="./log",
            ))
        
        # saving regularly in case
        with open(args.output, "wb") as f:
            pickle.dump(patients, f)

    print(f"{len(patients)} patients loaded")

    print("saving in pkl..")
    with open(args.output, "wb") as f:
        pickle.dump(patients, f)

    print("Done")
