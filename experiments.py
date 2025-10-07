import os, re, pickle, json, argparse, yaml
import pandas
import numpy as np
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale
from sklearn.utils.multiclass import unique_labels
from radiomics import getFeatureClasses

from dataloader import ARTIX, TCIA_HNSCC3DCTRT

def zero_division(num, den):
    try:
        if den == 0:
            return 0
        else:
            return num / den
    except TypeError:
        return 0


def build_dataset(internal_data, external_data, internal_features, external_features, combined_path):
    print("loading cohorts...")
    df = pandas.DataFrame([])
    for loader, data, features, cohort in ((ARTIX(), internal_data, internal_features, "internal"), 
                                           (TCIA_HNSCC3DCTRT(), external_data, external_features, "external")):
        
        features = pandas.read_csv(features)
        features["cohort"] = cohort
        
        with open(data, "rb") as f:
            patients = pickle.load(f)

        for p in patients:
            patient_df = []
            for k, v in loader.load_clinical(p).items():
                patient_df.append({"cohort": cohort, "features": "clinical", "name": k, "value": v, "id": p.id})

            features = pandas.concat((features, pandas.DataFrame(patient_df)))
        
        df = pandas.concat((df, features))

    print("saving dataset...")
    pandas.DataFrame(df).to_csv(combined_path, index=False)


def display_split_stats(split):
    stats = {j: list(split).count(j) for j in set(split)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(split))}%)")

def cross_validation(X_internal, Y_internal, X_external, Y_external, normalization="minmax", bootstrap=None, feature_selection=None, feature_reduction_N=None):
    """
    Perform boostrap training and testing and return classification metrics as dict

    args
        X_ (internal, external) (numpy.ndarray) data input of shape (n_samples, n_features)
        Y_ (internal, external) (numpy.ndarray) data expected output of shape (n_samples)
        normalization (str) method for normalizing input data
        bootstrap (int) number of time to eprform the training with random splitting, if None apply k-fold
        feature_selection (str) feature selection method to apply (see sklearn)
        feature_reduction_N (int) number of dimension to reduce data into using PCA
    """

    def compute_metrics(y_pred, y_pred_proba, y):
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        m = {"acc": accuracy_score(y, y_pred),
             "auc": roc_auc_score(y, y_pred),
             "balanced_accuracy": balanced_accuracy_score(y, y_pred),
             "f1_score": f1_score(y, y_pred, zero_division=0),
             "specificity": zero_division(tn, (tn + fp)),
             "sensitivity": zero_division(tp, (tp + fn)),
             "log_loss": log_loss(y, y_pred_proba)}
        return m

    print("Y internal stat:")
    print(display_split_stats(Y_internal))
    print("Y external stat:")
    print(display_split_stats(Y_external))

    if not bootstrap:
        bootstrap = 1

    metrics = []
    for i in tqdm.trange(bootstrap):
        if unique_labels(Y_internal).size == 1 or unique_labels(Y_external).size == 1:
            print("WARNING: skipping bootstrap step due to unique label in y_train or y_test")
            continue

        # normalize features separatly (to avoid data leakage)
        if normalization == "minmax":
            X_internal = minmax_scale(X_internal, axis=0)
            X_external = minmax_scale(X_external, axis=0)
        else:
            X_internal = scale(X_internal, axis=0)
            X_external = scale(X_external, axis=0)

        if feature_selection:
            raise NotImplementedError() #TODO

        if feature_reduction_N:
            if 0 < feature_reduction_N and feature_reduction_N < min(X_internal.shape):
                reduc = PCA(n_components=feature_reduction_N, random_state=0)
                reduc.fit(X_internal)
                X_internal = reduc.transform(X_internal)
                X_external = reduc.transform(X_external)
            else:
                # print(f"input cannot be transformed {X_internal.shape[1]} to {feature_reduction_N} dimensions since final dimension must be between 1 and min(n_samples, n_features)={min(X_internal.shape)}")
                pass

        try:
            clf = LogisticRegression(verbose=0)
            clf.fit(X_internal, Y_internal)

            y_pred_internal = clf.predict(X_internal)
            y_pred_proba_internal = clf.predict_proba(X_internal)

            y_pred_external = clf.predict(X_external)
            y_pred_proba_external = clf.predict_proba(X_external)

            internal_metrics = compute_metrics(y_pred_internal, y_pred_proba_internal, Y_internal)
            external_metrics = compute_metrics(y_pred_external, y_pred_proba_external, Y_external)
            metrics.append((internal_metrics, external_metrics))
        except ValueError:
            print("ValueError occured during model training or test")
            continue

    return metrics


def run_experiment(internal_data, external_data, internal_features, external_features, exps, exp_code_, output):
    """
    args
        #TODO
    """
    combined_path = "./dataset.csv"
    build_dataset(internal_data, external_data, internal_features, external_features, combined_path)
    df = pandas.read_csv(combined_path)
    df["id"] = df["id"].astype("str")

    print("filtering patients...")

    # filter patients without xerostomia label
    subdf = df[df["name"] == "xerostomia"]
    subdf = subdf[pandas.isna(subdf["value"])]
    df = df[~df["id"].isin(subdf["id"].unique())]
    print("number of patients remaning: ", len(df["id"].unique()))

    for i, exp_params in enumerate(exps):
        print(f"running experiment {i+1}/{len(exps)}")

        # create experience name
        exp_name = f"{exp_code_}_{str(i+1)}"

        # select data containing OARs; clinical features must be added because they would be lost after that
        # features filtering comes after
        if not isinstance(exp_params["oars"], (list, tuple)):
            exp_params["oars"] = [exp_params["oars"]]
        exp_df = df[(df["oar"].isin(exp_params["oars"])) | (df["features"] == "clinical")]

        # remove xerostomia information (target to predict)
        exp_df = exp_df.drop(exp_df[exp_df["name"] == "xerostomia"].index)

        # select features
        for fts, names in exp_params["features"].items():
            if names == -1:
                # keep all features belong to this type
                continue
            elif isinstance(names, list):
                    exp_df = exp_df.drop(exp_df[(exp_df["features"] == fts) & (~exp_df["name"].isin(names))].index)
            else:
                raise TypeError()

        # build X (input)
        X = exp_df.copy(deep=True)
        X['features'] = X[['features', 'name']].agg('_'.join, axis=1)
        X = X[["id", "features", "oar", "value", "cohort"]]

        internal_patient_id = X[X["cohort"] == "internal"]["id"].unique()
        external_patient_id = X[X["cohort"] == "external"]["id"].unique()

        # tranform values to float (handle nan)
        # merge OARs features
        point_float_pattern = r"-?\d+\.\d+|-?\d+"
        convert_float = lambda i: float(re.findall(point_float_pattern, i)[0])
        X["value"] = X["value"].apply(convert_float).astype("float32")
        square_mean = lambda x: np.sqrt(np.square(x).sum())
        X = X.groupby(["id", "features", "cohort"], as_index=False)["value"].apply(square_mean)

        # reshape dataframe and drop patients with nan values
        # TODO features completion
        X = X.pivot(index="id", columns="features", values="value")
        index_n_prev = len(X.index)
        X = X.dropna(axis="index")
        
        print("removing patients because missing OAR or feature: ", index_n_prev - len(X.index))
        print("number of patients remaning: ", len(X.index))

        # split into training and testing data
        internal_patient_id = [i for i in internal_patient_id if i in list(X.index.values)]
        external_patient_id = [i for i in external_patient_id if i in list(X.index.values)]
        X_internal = X.loc[internal_patient_id]
        X_external = X.loc[external_patient_id]

        # build Y
        # this must be done on original DataFrame because toxicity value is filtered in exp dataframe
        Y_internal = df[df["name"] == "xerostomia"].pivot(index="id", columns="name", values="value").loc[X_internal.index.values, "xerostomia"]
        Y_external = df[df["name"] == "xerostomia"].pivot(index="id", columns="name", values="value").loc[X_external.index.values, "xerostomia"]

        # convert to numpy arrays
        X_internal = np.array(X_internal)
        X_external = np.array(X_external, dtype=np.float32)
        Y_internal = np.array(Y_internal, dtype=np.float16).astype(dtype=np.int16)
        Y_external = np.array(Y_external, dtype=np.float16).astype(dtype=np.int16)

        print("X internal shape: ", X_internal.shape)
        print("Y internal shape: ", Y_internal.shape)
        print()
        print("X external shape: ", X_external.shape)
        print("Y external shape: ", Y_external.shape)

        # fit model
        print("fitting model")
        metrics = cross_validation(X_internal, Y_internal, X_external, Y_external, 
                                   normalization=exp_params["normalization"], bootstrap=exp_params["bootstrap"], 
                                   feature_reduction_N=exp_params["feature_reduction_N"])
        
        internal_metrics, external_metrics = zip(*metrics)

        # save results
        out_dir = os.path.join(output, exp_code_, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        pandas.DataFrame(internal_metrics).to_csv(os.path.join(out_dir, "internal_metrics.csv"))
        pandas.DataFrame(external_metrics).to_csv(os.path.join(out_dir, "external_metrics.csv"))

        # save exp params
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(exp_params, f)

def list_radiomics(type_):
    features = getFeatureClasses()[type_].getFeatureNames().keys()
    return [f"original_{type_}_{i}" for i in features]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--internal_data', type=str, required=True, help='path to internal data pickle file')
    parser.add_argument('--external_data', type=str, required=True, help='path to external data pickle file')
    parser.add_argument('--internal_features', type=str, required=True, help='path to internal data features (csv)')
    parser.add_argument('--external_features', type=str, required=True, help='path to external data features (csv)')
    parser.add_argument('--output', type=str, required=True, help='path to save results')
    parser.add_argument('--exp_yaml', type=str, required=True, help='path to yaml file containing experiment parameters')
    parser.add_argument('--feature_reduction_N', type=int, default=10, help='number of dimensions to reduce feature vector')
    parser.add_argument('--bootstrap', type=int, default=100, help='number of bootstraps')
    parser.add_argument('--normalization', type=str, default="minmax", help='normalization method to apply to features')
    args = parser.parse_args()
    
    list_of_oars = [("parotid_gland_left", "parotid_gland_right"), 
                    ("submandibular_gland_right", "submandibular_gland_left"), 
                    ("mandible")]
    
    with open(args.exp_yaml, 'r') as file:
        features = yaml.safe_load(file)

    exp_code = str(features["code"]).zfill(3)
    features = features["experiments"]

    exps = []
    for oars in list_of_oars:
        for fts in features:
            exp_params = dict(oars=oars, features=fts, feature_reduction_N=args.feature_reduction_N, 
                              bootstrap=args.bootstrap, normalization=args.normalization)
            exps.append(exp_params)
    run_experiment(args.internal_data, args.external_data, args.internal_features, args.external_features, exps, exp_code, output=args.output)
    print("done.")
