import math
from itertools import chain
import numpy as np
import pandas
import torch
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from datasets import ARTIX, HECKTOR, HeadNeckCTAtlas, HeadNeckPETCT, QINHEADNECK, OropharyngealRadiomicsOutcomes, RADCURE
from datasets import CATEGORICAL_CLINICAL_VARIABLES


class Normalizer:
    def __init__(self, method=None):
        match method:
            case "scale":
                self.normalizer = sklearn.preprocessing.StandardScaler()
            case "minmax":
                self.normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            case "unit":
                self.normalizer = sklearn.preprocessing.Normalizer()
            case _:
                self.normalizer = sklearn.preprocessing.StandardScaler()
    
    def fit_transform(self, X):
        X.loc[:,:] = self.normalizer.fit_transform(X.values)
        return X

    def transform(self, X):
        X.loc[:,:] = self.normalizer.transform(X.values)
        return X
    
    def get_params(self):
        return self.normalizer.__getstate__()

    def set_params(self, params):
        self.normalizer.__setstate__(params)

class MultiCenterStratifiedKFold:
    def __init__(self, Y):
        self.Y = Y

    def split(self, kfold=None, train_val_split=0.6, per_center=False):
        """
        Split data with stratified k-fold

        kfold (int) number of folds
        train_val_split (float) train/validation data factor for training, the rest is for validation
        per_center (bool) wether to split data seaparatly according to center
        """

        if kfold is None:
            if per_center:
                # if (all centers) size > 500 k=5 else k=3
                sizes = np.array([len(self.Y[self.Y["center"] == c]) for c in self.Y["center"].unique()])
                kfold = 5 if np.all(sizes >= 500) else 3
            else:
                kfold = 5 if len(self.Y) > 500 else 3

        if per_center:
            # define indices and labels
            sample_idx = {c: np.array(self.Y[self.Y["center"] == c].index) for c in self.Y["center"].unique()}
            sample_labels = {c: self.Y[self.Y["center"] == c]["label"] for c in self.Y["center"].unique()}

            # stratified kfold split to ensure similar label distribution across folds
            skf = {c: StratifiedKFold(n_splits=kfold, shuffle=True).split(sample_idx[c], sample_labels[c].values) for c in self.Y["center"].unique()}

            for _ in range(kfold):
                train_idx = {}
                val_idx = {}
                test_idx = {}
                for c, skfold in skf.items():
                    train_idx_center, test_idx_center = next(skfold)
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_val_split)
                    train_idx_center_, val_idx_center_ = next(sss.split(sample_idx[c][train_idx_center], sample_labels[c].loc[sample_idx[c][train_idx_center]]))
                    train_idx.update({c: sample_idx[c][train_idx_center][train_idx_center_]})
                    val_idx.update({c: sample_idx[c][train_idx_center][val_idx_center_]})
                    test_idx.update({c: sample_idx[c][test_idx_center]})
                
                # concatenate indices from different centers
                train_idx = np.concatenate(list(train_idx.values()))
                val_idx = np.concatenate(list(val_idx.values()))
                test_idx = np.concatenate(list(test_idx.values()))
                yield train_idx, val_idx, test_idx
            return StopIteration
        else:
            # define indices
            sample_idx = np.array(self.Y.index)

            # stratified kfold split to ensure similar label distribution across folds
            skf = StratifiedKFold(n_splits=kfold, shuffle=True)

            for train_idx, test_idx in skf.split(sample_idx, self.Y["label"].values):
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_val_split)
                train_idx_, val_idx_ = next(sss.split(sample_idx[train_idx], self.Y.loc[sample_idx[train_idx]]["label"].values))
                yield sample_idx[train_idx][train_idx_], sample_idx[train_idx][val_idx_], sample_idx[test_idx]
            return StopIteration


class DataLoader:
    def __init__(self, base_path, name, X=None, Y=None, uniform_sampling=False, class_weights=False):
        self.base_path = base_path
        self.name = name
        self.X = X
        self.Y = Y
        self.uniform_sampling = uniform_sampling
        self.class_weights = class_weights

        if class_weights and not(Y is None):
            self.cw = self.comput_class_weights()

    def len(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X.loc[idx], self.Y.loc[idx]

    def one_hot_encode(self, value, max_value):
        one_hot = np.zeros(int(max_value) + 1)
        one_hot[int(value)] = 1
        return one_hot

    def load(self):
        match self.name:
            case "artix":
                loader = ARTIX
            case "hecktor":
                loader = HECKTOR
            case "headneckctatlas":
                loader = HeadNeckCTAtlas
            case "headneckpetct":
                loader = HeadNeckPETCT
            case "qinheadneck":
                loader = QINHEADNECK
            case "oropharyngealradiomicsoutcomes":
                loader = OropharyngealRadiomicsOutcomes
            case "radcure":
                loader = RADCURE
            case _:
                raise ValueError(f"dataset name {self.name} not recognized. See args.dataset argument choices.")
            
        loader = loader(None)
        x, y = loader.get_features_labels(self.base_path)
        self.X = x
        self.Y = y

    def inclusion_criteria_clinical_features(self, row):
        """
        Return True if feature value corresponds to inclusion criteria, False otherwise
        """

        if row["value"] is None:
            return False

        match row["features"]:
            case "dose":
                return 64 <= row["value"] and row["value"] < 80
            case "metastasis":
                return row["value"] == 0
            case "localisation":
                return row["value"] == 0
            case "treatment":
                return row["value"] in [0,1]
            case "surgery":
                return row["value"] == 0
            case _:
                # if variable not in inclusion criteria list, include it anyways
                return True

    def preprocess(self, exp_params, task):
        """
        remove patients with missing data or invalid label
        prepare data for training according to experiment parameters and task
        keep only modalities and imaging features defined in exp_params
        """

        def row_to_label(row):
            if task == "rfs_2":
                T = 2
            elif task == "rfs_5":
                T = 5
            else:
                raise ValueError(f"task {task} not valid. Valid ones are [{"rfs_2"}, {"rfs_5"}]")

            if row["rfs_T"] is None:
                return None
            elif row["rfs_T"] < T*365:
                if row["rfs_delta"] == 1:
                    return 0
                else:
                    return None
            else:
                return 1

        # select only imaging features defined in exp_params
        self.X = self.X[(self.X["modality"] != "image") | ((self.X["modality"] == "image") & (self.X["features"].isin(exp_params["extractors"])))]

        # one hot encode categorical clinical variables
        X_clinical = self.X[self.X["modality"] == "clinical"]
        df = []
        for _, row in X_clinical.iterrows():
            if not(self.inclusion_criteria_clinical_features(row)):
                continue
            
            try:
                if row["features"] in CATEGORICAL_CLINICAL_VARIABLES:
                    max_value = X_clinical[X_clinical["features"] == row["features"]]["value"].max()
                    if max_value > 1:
                        one_hot = self.one_hot_encode(row["value"], max_value)
                        for i, v in enumerate(one_hot):
                            df.append({"patient": row["patient"], "modality": row["modality"], "features": row["features"], "name": i, "value": v})
                    else:
                        df.append(row.to_dict())
                else:
                    df.append(row.to_dict())
            except (ValueError, TypeError):
                continue
        
        X_clinical = pandas.DataFrame(df)
        X_image = self.X[self.X["modality"] != "clinical"]

        # merge modalities
        if exp_params["modality"] == "image":
            self.X = X_image
        elif exp_params["modality"] == "clinical":
            self.X = X_clinical
        else:
            self.X = pandas.concat([X_image, X_clinical], ignore_index=True)

        # pivot dataframe and sort columns names
        self.X = self.X.pivot(index="patient", columns=["modality", "features", "name"], values="value")
        self.X = self.X.sort_index(axis="columns")

        # set patient as index for labels dataframe and compute label column according to task
        self.Y = self.Y.set_index("patient")        
        self.Y["label"] = self.Y.apply(row_to_label, axis=1)

        # remove patients with invalid label, missing data or uncommon between X and Y
        invalid_label_patients = self.Y[self.Y["label"].isna()].index
        missing_data_patients = self.X[self.X.isna().any(axis=1)].index
        uncommon_patients = [
            *[p for p in self.X.index if not(p in self.Y.index)],
            *[p for p in self.Y.index if not(p in self.X.index)]
        ]
        remove_indices = [*invalid_label_patients, *missing_data_patients, *uncommon_patients]

        # patients that are filtered but should be preserved for Cox model training 
        # (those with missing label but available RFS time)
        preserve_Cox_idx = self.Y[(self.Y["label"].isna()) & (self.Y["rfs_T"] != None)].index
        preserve_Cox_idx = [i for i in preserve_Cox_idx if not(i in [*missing_data_patients, *uncommon_patients])]
        self.Cox_X = self.X.loc[preserve_Cox_idx]
        self.Cox_Y = self.Y.loc[preserve_Cox_idx]

        # remove filtered patients from dataset
        self.X = self.X.drop(index=remove_indices, errors="ignore")
        self.Y = self.Y.drop(index=remove_indices, errors="ignore")

        # if using loss class weighting, compute class weights
        if self.class_weights:
            self.cw = self.comput_class_weights()

    def comput_class_weights(self, t=0.01):
        if self.Y is None:
            raise ValueError("self.Y must be initialized before computing class weights")
        
        freq = dict(self.Y['label'].value_counts())   # frequence dict
        cw = torch.zeros(len(freq), dtype=torch.float32)
        for i in freq.keys():   # populate values with inverse frequence
            i = int(i)
            cw[i] = 1 / freq[i]
        cw = torch.softmax(cw / t, dim=0)   # apply softmax with temperature to increase heterogeneity
        cw = (cw - cw.mean()) + 1   # center values around 1
        return cw

    def split(self, kfold=None, train_val_split=0.6, per_center=False):
        """
        split data into kfold train/validation/test sets
        
        kfold (int) number of folds
        train_val_split (float) proportion of training data used for training (rest for validation)
        per_center (bool) wether to split data seaparatly according to center
        """

        splitter = MultiCenterStratifiedKFold(self.Y)
        for train_idx, val_idx, test_idx in splitter.split(kfold, train_val_split, per_center):
            # (train data), (val data), (test data)
            yield (self.X.loc[train_idx], self.Y.loc[train_idx]), (self.X.loc[val_idx], self.Y.loc[val_idx]), (self.X.loc[test_idx], self.Y.loc[test_idx])
        return StopIteration

    def undersampling(self, per_center=False):
        if per_center:
            idx = []
            for center in self.Y["center"].unique():
                label_counts = self.Y[self.Y["center"] == center]['label'].value_counts()
                n = min(label_counts.values)
                if n == 0:
                    idx.append(self.Y[self.Y["center"] == center].index)
                else:
                    for i in label_counts.keys():
                        idx.append(np.random.choice(self.Y[(self.Y["center"] == center) & (self.Y["label"] == i)].index, size=n, replace=False))
            idx = np.concatenate(idx, axis=0)
        else:
            label_counts = self.Y['label'].value_counts()
            n = min(label_counts.values)
            idx = [np.random.choice(self.Y[self.Y["label"] == i].index, size=n, replace=False) for i in label_counts.keys()]
            idx = np.concatenate(idx, axis=0)
        self.X = self.X.loc[idx]
        self.Y = self.Y.loc[idx]

    def normalize_image_features(self, normalizer=None):
        if isinstance(normalizer, Normalizer):
            self.X.loc[:, ["image"]] = normalizer.transform(self.X.loc[:, ["image"]])
        else:
            normalizer = Normalizer(normalizer)
            self.X.loc[:, ["image"]] = normalizer.fit_transform(self.X.loc[:, ["image"]])
            return normalizer
        
    def normalize_clinical_features(self):
        if "age" in self.X.columns.get_level_values("features"):
            self.X.loc[:, ("clinical", "age")] = self.X.loc[:, ("clinical", "age")].values / 100.
        if "dose" in self.X.columns.get_level_values("features"):
            self.X.loc[:, ("clinical", "dose")] = self.X.loc[:, ("clinical", "dose")].values / 70.

    def get_random_batch(self, n, sample_weight=False):
        """
        get random batch taking into account centers equality
        """
        if self.uniform_sampling:
            centers = self.Y["center"].unique()
            n_per_center = math.ceil(n / len(centers))
            idx = []
            for c in centers:
                indices = self.Y[self.Y["center"] == c].index
                # take random sample as min between center size and batch/centers, otherwise error of sampling
                idx.append(np.random.choice(indices, size=min(len(indices), n_per_center), replace=False))
            idx = list(chain.from_iterable(idx))[:n]
        else:
            idx = np.random.choice(self.X.index, size=n, replace=False)
        
        x = self.X.loc[idx]
        y = self.Y.loc[idx]

        x = {m: {
                k: torch.tensor(x[(m, k)].values, dtype=torch.float32) for k in x.columns.get_level_values("features")[x.columns.get_level_values("modality") == m].unique()
            } for m in x.columns.get_level_values("modality").unique()}
        y = torch.tensor(y["label"].values)

        if sample_weight:
            cw = self.cw[y.to(dtype=torch.long)]
            return x, y, cw
        else:
            return x, y
    
    def get_random_batch_posneg(self, n):
        x_pos = self.X.loc[np.random.choice(self.Y[self.Y["label"] == 1].index, size=n)]
        x_neg = self.X.loc[np.random.choice(self.Y[self.Y["label"] == 0].index, size=n)]

        x_pos = {m: {
                k: torch.tensor(x_pos[(m, k)].values, dtype=torch.float32) for k in x_pos.columns.get_level_values("features")[x_pos.columns.get_level_values("modality") == m].unique()
            } for m in x_pos.columns.get_level_values("modality").unique()}
        
        x_neg = {m: {
                k: torch.tensor(x_neg[(m, k)].values, dtype=torch.float32) for k in x_neg.columns.get_level_values("features")[x_neg.columns.get_level_values("modality") == m].unique()
            } for m in x_neg.columns.get_level_values("modality").unique()}
        
        return x_pos, x_neg
    
    def batch_iterator(self, batch_size, sample_weight=False):
        for idx in range(0, len(self.Y), batch_size):   # handle case where dataset smaller than batch size
            indices = self.Y.index[idx:idx+batch_size]
            x = self.X.loc[indices]
            y = self.Y.loc[indices]

            x_ = {}
            for m in x.columns.get_level_values("modality").unique():
                if m == "image":
                    x_.update({m: {k: torch.tensor(x[(m, k)].values, dtype=torch.float32) for k in x.columns.get_level_values("features")[x.columns.get_level_values("modality") == m].unique()}})
                else:
                    x_.update({m: torch.tensor(x[(m)].values, dtype=torch.float32)})
            x = x_
            
            y = torch.tensor(y["label"].values)

            if sample_weight:
                cw = self.cw[y.to(dtype=torch.long)]
            else:
                cw = None

            yield x, y, cw
        return StopIteration


class CoxLoader:
    def __init__(self, X, Y, strategy="1v1"):
        self.X = X
        self.Y = Y
        self.strategy = strategy

    def __len__(self):
        if self.strategy == "1v1":
            return len(self.neg_idx)
        else:
            return len(self.valid_idx)

    def prepare_data(self, task=None):
        if self.strategy == "1v1" and task is None:
            raise ValueError("task argument must be provided for 1v1 strategy")
        
        if self.strategy == "1v1":
            if task == "rfs_2":
                T = 2
            elif task == "rfs_5":
                T = 5
            else:
                raise ValueError(f"task {task} not valid. Valid ones are [{"rfs_2"}, {"rfs_5"}]")
            
            self.pos_idx = self.Y[self.Y["rfs_T"] >= T*365].index
            self.neg_idx = self.Y[(self.Y["rfs_T"] < T*365) & (self.Y["rfs_delta"] == 1)].index

            if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
                raise ValueError(f"No positive or negative sample for Cox pre-training with {task} task. Consider changing task or strategy.")
            
            if len(self.pos_idx) != len(self.neg_idx):
                n = min(len(self.pos_idx), len(self.neg_idx))
                self.pos_idx = np.random.choice(self.pos_idx, size=n, replace=False)
                self.neg_idx = np.random.choice(self.neg_idx, size=n, replace=False)
        else:
            self.valid_idx = self.Y[self.Y["rfs_delta"] == 1].index        

    def batch_iterator(self, batch_size, skip_singleton_batch=True):
        def dataframe_to_tensor(x):
            x_ = {}
            for m in x.columns.get_level_values("modality").unique():
                if m == "image":
                    x_.update({m: {k: torch.tensor(x[(m, k)].values, dtype=torch.float32) for k in x.columns.get_level_values("features")[x.columns.get_level_values("modality") == m].unique()}})
                else:
                    x_.update({m: torch.tensor(x[(m)].values, dtype=torch.float32)})
            return x_
        
        if self.strategy == "1v1":
            self.pos_idx = np.random.permutation(self.pos_idx)
            self.neg_idx = np.random.permutation(self.neg_idx)
            pairs = list(zip(self.pos_idx, self.neg_idx))
            for idx in range(0, len(pairs), batch_size):
                pos, neg = zip(*pairs[idx : idx + batch_size])
                neg, pos = tuple(neg), tuple(pos)
                if skip_singleton_batch and len(neg) == 1:
                    return StopIteration   # stop if batch size is 1 (BatchNorm error)
                neg = dataframe_to_tensor(self.X.loc[neg])
                pos = dataframe_to_tensor(self.X.loc[pos])
                yield neg, pos
        else:
            self.valid_idx = np.random.permutation(self.valid_idx)
            for idx in range(0, len(self.valid_idx), batch_size):
                idx = self.valid_idx[idx : idx + batch_size]
                if skip_singleton_batch and len(idx) == 1:
                    return StopIteration   # stop if batch size is 1 (BatchNorm error)
                neg = dataframe_to_tensor(self.X.loc[idx])
                pos = []
                for i in idx:
                    pos_idx_i = self.Y[self.Y["rfs_T"] > self.Y.loc[i, "rfs_T"]].index
                    pos.append(dataframe_to_tensor(self.X.loc[pos_idx_i]))
                yield neg, pos
        return StopIteration
