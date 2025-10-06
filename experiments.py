import os, itertools, re, pickle, json, copy
import pandas
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale
from sklearn.utils.multiclass import unique_labels
import numpy as np
import tqdm
from radiomics import getFeatureClasses

def zero_division(num, den):
    try:
        if den == 0:
            return 0
        else:
            return num / den
    except TypeError:
        return 0


def transform_(x):
    if isinstance(x, str):
        digits = re.findall("\d", x)
        if len(digits) == 0:
            return None
        else:
            return int(digits[0])
    else:
        return None


def load_features(path_, type_, patient_id, columns_to_keep=["oar", "name", "value"]):
    """
    load features

    args:
        path_ (str) path to features folder
        type_ (str) features type (radiomics, dosiomics, dvh, deepnn)
        patient_id (int,str) id of patient to find its folder
        columns_to_keep (list(str)) list of columns name to keep
    """
    patient_folder = os.path.join(path_, [i for i in os.listdir(path_) if int(i) == int(patient_id)][0])
    features = pandas.read_csv(os.path.join(patient_folder, f"{type_}.csv"))

    if "Unnamed: 0" in features.columns:
        features = features.drop(columns=["Unnamed: 0"])

    if columns_to_keep:
        features = features[columns_to_keep]

    return features

def load_radiomics(path_, patient_id, omics):
    """
    omics (radiomics, dosiomics)
    """
    assert omics in ["radiomics", "dosiomics"]
    df = load_features(path_, omics, patient_id, columns_to_keep=None)
    df = df[(df["type"] == "original")]
    df['name'] = df[["type", "class", "name"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df.drop(columns=["type", "class"])


def build_dataset(cohort_path, features_path, combined_path):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! adapt this for TCIA 
    clinical = ['SEX', 'AGE', 'ST_STAG', 'ECOG']
    CLINICAL_MAPPING = {
        # SEX
        "Male": 1,
        "Female": 2,
        # "M": 1,
        # "F": 2,

        # CANCER STAGING
        "Stage III": 1,
        "Stage IVa": 2,
        "Stage IVb": 3,

        # ECOG
        "Asympatomatic": 0,
        "Completely ambulatory": 1,
        "lower than 50% in bed": 2,
    }

    print("loading cohort...")
    with open(cohort_path, "rb") as f:
        patients = pickle.load(f)

    print("building dataset...")
    df = []
    for p in patients:
        patient_df = []

        for k in clinical:
            try:
                patient_df.append({"features": "clinical", "name": k, "value": int(p.clinical[k]), "id": int(p.id)})
            except ValueError:
                patient_df.append({"features": "clinical", "name": k, "value": CLINICAL_MAPPING[p.clinical[k]], "id": int(p.id)})

        # select xerostomia in CTCAE
        # transform values
        # change S0 into Inclusion
        # if patient has baseline tox, skip it
        # keep only M6 grade
        cm = pandas.DataFrame(p.clinical_measurements)
        cm = cm[(cm["type"] == "AE") & (cm["sample"] == "XEROSTOMIE")]
        cm["value"] = cm["value"].apply(transform_)
        cm.loc[(cm["visitID"] == "S0"), "visitID"] = "Inclusion"
        if (cm[(cm["visitID"] == "Inclusion") & (cm["sample"] == "XEROSTOMIE")]["value"].astype('Int64') > 0).any():
            continue
        cm = cm[cm["visitID"] == "M6"]

        # transform tox value to binary
        cm.loc[(cm["value"] < 2), "value"] = 0
        cm.loc[(cm["value"] >= 2), "value"] = 1

        for _, row in cm.iterrows():
            patient_df.append({"features": "clinical", "name": row["sample"], "value": row["value"], "id": int(p.id)})

        try:
            # radiomics
            radiomics = load_radiomics(features_path, p.id, "radiomics")
            radiomics["id"] = int(p.id)
            radiomics["features"] = "radiomics"
            patient_df.extend(radiomics.to_dict(orient="records"))
        except (FileNotFoundError, IndexError):
            print(f"WARNING: patient {p.id} missing radiomics data")
            continue

        try:
            # dosiomics
            dosiomics = load_radiomics(features_path, p.id, "dosiomics")
            dosiomics["id"] = int(p.id)
            dosiomics["features"] = "dosiomics"
            patient_df.extend(dosiomics.to_dict(orient="records"))
        except (FileNotFoundError, IndexError):
            print(f"WARNING: patient {p.id} missing dosiomics data")
            continue

        try:
            # dvh
            dvh = load_features(features_path, "dvh", p.id)
            dvh["id"] = int(p.id)
            dvh["features"] = "dvh"
            patient_df.extend(dvh.to_dict(orient="records"))
        except (FileNotFoundError, IndexError):
            print(f"WARNING: patient {p.id} missing dvh data")
            continue

        try:
            # deepnn
            for name_ in ("deepnn", "deepnn(fmcib)"):
                deepnn = load_features(features_path, name_, p.id)
                deepnn["id"] = int(p.id)
                deepnn["features"] = name_
                patient_df.extend(deepnn.to_dict(orient="records"))
        except (FileNotFoundError, IndexError):
            print(f"WARNING: patient {p.id} missing deepnn data")
            continue

        df.extend(patient_df)

    print("saving dataset...")
    pandas.DataFrame(df).to_csv(combined_path, index=False)


def display_split_stats(split):
    stats = {j: list(split).count(j) for j in set(split)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(split))}%)")

def cross_validation(X, Y, normalization="minmax", kfold=3, bootstrap=None, feature_selection=None, feature_reduction_N=None, verbose=False):
    """
    Perform cross validation training and return classification metrics as dict

    args
        X (numpy.ndarray) training data input of shape (n_samples, n_features)
        Y (numpy.ndarray) training data expected output of shape (n_samples)
        normalization (str) method for normalizing input data
        kfold (int) number of folds default is 3
        bootstrap (int) number of time to eprform the training with random splitting, if None apply k-fold
        feature_selection (str) feature selection method to apply (see sklearn)
        feature_reduction_N (int) number of dimension to reduce data into using PCA
    """

    if bootstrap:
        splitter = ShuffleSplit(n_splits=bootstrap, random_state=0, test_size=0.3)
    else:
        splitter = KFold(n_splits=kfold)

    metrics = []
    reductor = []
    for i, (train_index, test_index) in enumerate(splitter.split(X)):
        x_train, y_train = X[train_index], Y[train_index]
        x_test, y_test = X[test_index], Y[test_index]

        if unique_labels(y_train).size == 1 or unique_labels(y_test).size == 1:
            print("WARNING: skipping bootstrap step due to unique label in y_train or y_test")
            continue

        # normalize features separatly (to avoid data leakage)
        if normalization == "minmax":
            x_train = minmax_scale(x_train, axis=0)
            x_test = minmax_scale(x_test, axis=0)
        else:
            x_train = scale(x_train, axis=0)
            x_test = scale(x_test, axis=0)

        if verbose:
            if bootstrap:
                print("bootstrap ", i+1)
            else:
                print("fold ", i+1)
            print("train:")
            display_split_stats(y_train)
            print("test:")
            display_split_stats(y_test)

        if feature_selection:
            raise NotImplementedError() #TODO

        if feature_reduction_N:
            if 0 < feature_reduction_N and feature_reduction_N < min(X.shape):
                if verbose:
                    print(f"transforming input from {x_train.shape[1]} to {feature_reduction_N} dimensions..")
                reduc = PCA(n_components=feature_reduction_N, random_state=0)
                reduc.fit(x_train)
                x_train = reduc.transform(x_train)
                x_test = reduc.transform(x_test)
                reductor.append(reduc)
            else:
                if verbose:
                    print(f"input cannot be transformed {x_train.shape[1]} to {feature_reduction_N} dimensions since final dimension must be between 0 and min(n_samples, n_features)={min(X.shape)}")

        try:
            clf = LogisticRegression(random_state=0, verbose=0)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            metrics.append({
                "acc": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "specificity": zero_division(tn, (tn + fp)),
                "sensitivity": zero_division(tp, (tp + fn)),
                "log_loss": log_loss(y_test, y_pred_proba),
            })
        except ValueError:
            print("ValueError occured during model training or test")
            continue

    return metrics, reductor


def run_experiment(cohort_path, features_path, exps, exp_code, verbose=False):
    """
    Train LR model for classification

    args
        cohort_path (str) path to cohort pickle file
        features_path (str) path to folder containing features
        exps (List(dict)) list of dictionnaries containing experiments parameters
        exp_code (str) if of experiment as str
        verbose (bool) weither to print informations
    """
    combined_path = "./dataset.csv"
    build_dataset(cohort_path, features_path, combined_path)
    df = pandas.read_csv(combined_path)

    if verbose:
        print("filtering patients...")

    # filter patients without xerostomia label
    # add patients without any XEROSTOMIA line
    subdf = df[df["name"] == "XEROSTOMIE"]
    subdf = subdf[pandas.isna(subdf["value"])]
    patients_to_exclude = list(subdf["id"].unique())
    subdf = df[df["name"] == "XEROSTOMIE"]
    patients_to_exclude.extend(list(set(set(df["id"].unique()).difference(subdf["id"].unique()))))
    df = df[~df["id"].isin(patients_to_exclude)]
    if verbose:
        print("patients excluded due to absent xerostomia label: ", len(patients_to_exclude))
        print("number of patients remaning: ", len(df["id"].unique()))

    for i, exp_params in enumerate(tqdm.tqdm(exps)):
        # create experience name
        exp_name = f"{exp_code}_{str(i+1)}"

        # select data containing OARs; clinical features must be added because they would be lost after that
        # features filtering comes after
        # remove xerostomia information (target to predict)
        exp_df = df[(df["oar"].isin(exp_params["oars"])) | (df["features"] == "clinical")]
        exp_df = exp_df.drop(exp_df[exp_df["name"] == "XEROSTOMIE"].index)

        # select features
        for fts, names in exp_params["features"].items():
            if names == -1:
                # keep all features belong to this type
                continue
            elif isinstance(names, list):
                    exp_df = exp_df.drop(exp_df[(exp_df["features"] == fts) & (~exp_df["name"].isin(names))].index)
            else:
                raise TypeError()

        # build input
        X = exp_df.copy()
        X['features'] = X[['features', 'name']].agg('_'.join, axis=1)
        X = X.pivot(index="id", columns=["oar", "features"], values="value")
        features = list(X.columns.values)  # features names must be saved in JSON to recover the order later
        index_n_prev = len(X.index)
        X = X.dropna(axis="index")
        patients = X.index.values   # IMPORTANT: to build Y
        if verbose:
            print("removing patients because missing OAR or feature: ", index_n_prev - len(X.index))
            print("number of patients remaning: ", len(X.index))
            print("features: ", features)

        # build Y
        # !!! this must be done on original DataFrame because toxicity value is filtered in exp dataframe
        Y = df[df["name"] == "XEROSTOMIE"].pivot(index="id", columns="name", values="value").loc[patients, "XEROSTOMIE"]

        # convert to numpy arrays
        try:
            X = np.array(X, dtype=np.float32)
        except ValueError:
            # in case values in X are not composed of numbers only
            f = lambda i: re.findall("\d+", i)[0]
            X = np.vectorize(f)(X).reshape(X.shape)
            X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float16).astype(dtype=np.int16)

        if verbose:
            print("X shape: ", X.shape)
            print("Y shape: ", Y.shape)

        # fit model
        if verbose:
            print("fitting model")

        if exp_params["bootstrap"]:
            metrics, _ = cross_validation(X, Y, normalization=exp_params["normalization"], bootstrap=exp_params["bootstrap"], feature_reduction_N=exp_params["feature_reduction_N"], verbose=verbose)
        else:
            metrics, _ = cross_validation(X, Y, normalization=exp_params["normalization"], kfold=exp_params["kfold"], feature_reduction_N=exp_params["feature_reduction_N"], verbose=verbose)

        # save results
        out_dir = os.path.join("./experiments", exp_code, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        pandas.DataFrame(metrics).to_csv(os.path.join(out_dir, "metrics.csv"))

        # save exp params
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            save_params = {**exp_params, "features_ordered": features}
            json.dump(save_params, f)


def list_radiomics(type_):
    features = getFeatureClasses()[type_].getFeatureNames().keys()
    return [f"original_{type_}_{i}" for i in features]


if __name__ == "__main__":
    OARS = {"original": ["parotid_gland_ipsi", "parotid_gland_contra", "submandibular_gland_ipsi", "submandibular_gland_contra", "mandible", "oral_cavity"],
            "totalseg": ["parotid_gland_left", "parotid_gland_right", "submandibular_gland_right", "submandibular_gland_left", "mandible"]}

    exp_code = "013"
    cohort_path = r"C:\Users\bilel.guetarni\Desktop\ARTIX\data\artix.pkl"

    base_params = {"oars": [], "features": {"clinical": [], "radiomics": [], "dosiomics": [], "dvh": [], "deepnn": [], "deepnn(fmcib)": -1},
                "feature_reduction_N": None, "normalization": "minmax", "bootstrap": 100, "kfold": 3}
    
    i = 0
    for features_path, seg_origin in [(r"C:\Users\bilel.guetarni\Desktop\ARTIX\features\original\artix", "original"),
                                      (r"C:\Users\bilel.guetarni\Desktop\ARTIX\features\totalsegmentator\artix", "totalseg")]:
        print(seg_origin)
        exps = []
        for oars in OARS[seg_origin]:
            for feature_reduction_N in [None, 10, 50, 100]:
                for features in [
                    {},
                    {"dvh": ["mean"]},
                ]:
                    base_params_copy = copy.deepcopy(base_params)
                    base_params_copy["oars"] = [oars]
                    base_params_copy["feature_reduction_N"] = feature_reduction_N
                    for k, v in features.items():
                        base_params_copy["features"][k] = v
                    exps.append(base_params_copy)
        exp_code_ = f"{exp_code}_{str(i).zfill(3)}"
        run_experiment(cohort_path, features_path, exps, exp_code_, verbose=False)
        i += 1
    print("Done.")
