import glob, os, tqdm, argparse, pickle
import artix

# path = r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data"
# id_map=r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\ARTIX_ID_CORRELATION.xlsx",
# clinical_csv=[
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_PATIENT_DESCRIPTION_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_EFFICACY_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_SALIVATION_DATA_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_TREATMENT_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_DOSIMETRY_LTSI.csv",
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_MDA_LTSI.csv"
#         r"C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data\20241021_AE_TOX_GEN_LTSI.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to ARTIX patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--id_map', type=str, required=True, help='path to xlsx file with ID mapping correlation')
    parser.add_argument('--save_freq', type=int, default=1, help='frequency (no of patients) to save')

    # clinical csv files
    parser.add_argument('--PATIENT_DESCRIPTION', type=str, help='path to PATIENT_DESCRIPTION csv')
    parser.add_argument('--EFFICACY', type=str, help='path to EFFICACY csv')
    parser.add_argument('--SALIVATION_DATA', type=str, help='path to SALIVATION_DATA csv')
    parser.add_argument('--TREATMENT', type=str, help='path to TREATMENT csv')
    parser.add_argument('--DOSIMETRY', type=str, help='path to DOSIMETRY csv')
    parser.add_argument('--MDA', type=str, help='path to MDA csv')
    parser.add_argument('--AETOXGEN', type=str, help='path to AE_TOX_GEN csv')
    args = parser.parse_args()

    patients = []
    for i, p in tqdm.tqdm(enumerate(glob.glob(os.path.join(args.input, "*"))), ncols=50):
        p = artix.load_patient(
            path=p,
            id_map=args.id_map,
            PATIENT_DESCRIPTION_csv=args.PATIENT_DESCRIPTION,
            EFFICACY_csv=args.EFFICACY,
            SALIVATION_DATA_csv=args.SALIVATION_DATA,
            TREATMENT_csv=args.TREATMENT,
            DOSIMETRY_csv=args.DOSIMETRY,
            MDA_csv=args.MDA,
            AETOXGEN_csv=args.AETOXGEN,
            log="./log",
            )
        
        # set base path
        p.update_study_base_path(args.input)

        patients.append(p)
        
        if ((i+1) % args.save_freq) == 0:
            print("saving in pkl..")
            with open(args.output, "wb") as f:
                pickle.dump(patients, f)

    print(f"\n {len(patients)} patients loaded")

    print("saving in pkl..")
    with open(args.output, "wb") as f:
        pickle.dump(patients, f)

    print("Done")
