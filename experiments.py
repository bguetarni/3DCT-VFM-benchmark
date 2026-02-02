import json
import pickle
import math
import argparse
import pathlib
import os
from itertools import chain
import pandas
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn.functional as F
import coolname

from dataloader import ARTIX, HECKTOR, HeadNeckCTAtlas, HeadNeckPETCT, QINHEADNECK, OropharyngealRadiomicsOutcomes, RADCURE
from dataloader import recurrence_2y_name, recurrence_5y_name, CATEGORICAL_CLINICAL_VARIABLES
from classifiers.classifiers import Attention, Linear, Concat, MLP_head, Normalizer

def one_hot_encode(value, max_value):
    one_hot = np.zeros(int(max_value) + 1)
    one_hot[int(value)] = 1
    return one_hot

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

    def build_dataset(self):
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
        x, y = loader.build_dataset(self.base_path)
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

    def prepare_data(self, exp_params, task):
        """
        remove patients with missing data or invalid label
        prepare data for training according to experiment parameters and task
        keep only modalities and imaging features defined in exp_params
        """

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
                        one_hot = one_hot_encode(row["value"], max_value)
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

        self.Y = self.Y.set_index("patient")
        if not(task in self.Y.columns):
            raise ValueError(f"task {task} not valid. Valid ones are [{recurrence_2y_name}, {recurrence_5y_name}]")
        self.Y = self.Y[["center", task]].rename(columns={task: "label"})

        # remove patients with invalid label, missing data or uncommon between X and Y
        invalid_label_patients = self.Y[self.Y["label"].isna()].index
        missing_data_patients = self.X[self.X.isna().any(axis=1)].index
        uncommon_patients = [
            *[p for p in self.X.index if not(p in self.Y.index)],
            *[p for p in self.Y.index if not(p in self.X.index)]
        ]
        remove_indices = [*invalid_label_patients, *missing_data_patients, *uncommon_patients]
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
    
    def batch_iterator(self, n, sample_weight=False):
        for idx in range(0, len(self.Y), n):   # handle case where dataset smaller than batch size
            indices = self.Y.index[idx:idx+n]
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


def zero_division(num, den):
    try:
        if den == 0:
            return 0
        else:
            return num / den
    except TypeError:
        return 0

def display_split_stats(loader):
    y = list(loader.Y["label"].values)
    stats = {int(j): y.count(j) for j in set(y)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(y))}%)")

def send_to_device(x, device):
    if isinstance(x, dict):
        return {k: send_to_device(v, device) for k,v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x).to(device)

def dkl_reg_loss(std_p, std_q):
    # see https://doi.org/10.1016/j.eswa.2021.115974
    std_p = std_p ** 2
    std_q = std_q ** 2
    return 0.5 * (torch.log(std_q / std_p) + (std_p / std_q) - 1)

def compute_metrics(y_pred, y_pred_proba, y):
    cm = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

    # handle case where only one label in y and y_pred
    # set all confusion matrix element to zero
    # keeping only the element relative to class present in y
    if len(cm) == 1:
        count = cm[0]
        cm = np.zeros((2,2), dtype=np.int64)
        i = int(np.unique(y).item())
        j = int(np.unique(y_pred).item())
        cm[i,j] = count
        cm = cm.ravel()

    tn, fp, fn, tp = cm
    m = {"acc": accuracy_score(y, y_pred),
            "auc": roc_auc_score(y, y_pred_proba),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "specificity": zero_division(tn, (tn + fp)),
            "sensitivity": zero_division(tp, (tp + fn)),
            "log_loss": log_loss(y, y_pred_proba, labels=[0,1])}
    return m

def eval(model, loader, batch_size, device):
    y_pred_proba = []
    y_pred = []
    y_true = []
    for batch in loader.batch_iterator(batch_size):
        if batch is StopIteration:
            break
        x, y, _ = batch
        with torch.no_grad():
            pred_proba = model(send_to_device(x, device)).to("cpu")
            pred_proba = F.sigmoid(pred_proba)
            y_pred_proba.append(pred_proba.flatten())
            y_pred.append(torch.round(pred_proba.flatten()))
            y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_true = torch.cat(y_true, dim=0)
    return compute_metrics(y_pred, y_pred_proba, y_true)

def kfold_training(exp_params, data_loader, kfold, device="cpu"):
    """
    Perform kfold training and testing and return classification metrics as dict

    Args:
        exp_params (dict) experiment parameters
        data_loader (DataLoader) data loader
        device (torch.device) device for model
        kfold (int) number of kfold cross-validation folds
    """

    if exp_params["undersampling"]:
        print("applying undersampling")
        data_loader.undersampling(per_center=args.per_center)

    train_metrics = []
    test_metrics = []
    best_state_dict = {}
    normalizer = {}
    patients_idxes = {}
    for k, (tain_data, valid_data, test_data) in enumerate(data_loader.split(kfold, train_val_split=args.train_split, per_center=args.per_center)):
        print(f"kfold {k+1}/{kfold}")
        X_train, Y_train = tain_data
        X_valid, Y_valid = valid_data
        X_test, Y_test = test_data

        patients_idxes.update({f"fold {k}": {
            "train": X_train.index.to_list(), 
            "valid": X_valid.index.to_list(), 
            "test": X_test.index.to_list()}})

        # construct dict of number of dimensions for each modality and features
        dims = {}
        for m, f, _ in X_train.columns:
            if m == "clinical":
                if not("clinical" in dims.keys()): dims.update({"clinical": 0})
                dims["clinical"] += 1
            else:
                if not(m in dims.keys()): dims.update({m: {}})
                if not(f in dims[m].keys()): dims[m].update({f: 0})
                dims[m][f] += 1
        exp_params.update({"dims": dims})

        # apply normalization function column-wise to image features
        print("normalizing features values..")
        norm = Normalizer(args.normalizer)
        if "image" in X_train.columns.get_level_values("modality"):
            X_train.loc[:, ["image"]] = norm.fit_transform(X_train.loc[:, ["image"]])
            X_valid.loc[:, ["image"]] = norm.transform(X_valid.loc[:, ["image"]])
            X_test.loc[:, ["image"]] = norm.transform(X_test.loc[:, ["image"]])
            normalizer.update({k: norm.get_params()})

        # normalize age
        if "age" in X_train.columns.get_level_values("features"):
            X_train.loc[:, ("clinical", "age")] = X_train.loc[:, ("clinical", "age")].values / 100.
            X_valid.loc[:, ("clinical", "age")] = X_valid.loc[:, ("clinical", "age")].values / 100.
            X_test.loc[:, ("clinical", "age")] = X_test.loc[:, ("clinical", "age")].values / 100.

        # normalize dose
        if "dose" in X_train.columns.get_level_values("features"):
            X_train.loc[:, ("clinical", "dose")] = X_train.loc[:, ("clinical", "dose")].values / 70.
            X_valid.loc[:, ("clinical", "dose")] = X_valid.loc[:, ("clinical", "dose")].values / 70.
            X_test.loc[:, ("clinical", "dose")] = X_test.loc[:, ("clinical", "dose")].values / 70.

        train_loader = DataLoader(base_path=None, name=None, X=X_train, Y=Y_train, uniform_sampling=exp_params["uniform_sampling"], class_weights=exp_params["class_weights"])
        valid_loader = DataLoader(base_path=None, name=None, X=X_valid, Y=Y_valid)
        test_loader = DataLoader(base_path=None, name=None, X=X_test, Y=Y_test)

        print("training data stat:")
        print(display_split_stats(train_loader))
        print("validation data stat:")
        print(display_split_stats(valid_loader))
        print("testing data stat:")
        print(display_split_stats(test_loader))

        match exp_params["classifier"]:
            case "attention":
                model = Attention(**exp_params)
            case "concat":
                model = Concat(**exp_params)
            case "linear":
                if isinstance(list(dims.values())[0], dict):
                    n_dim = sum(list(list(dims.values())[0].values()))
                else:
                    n_dim = list(dims.values())[0]
                model = Linear(n_dim, n_class=1)
            case "ffn":
                if isinstance(list(dims.values())[0], dict):
                    n_dim = sum(list(list(dims.values())[0].values()))
                else:
                    n_dim = list(dims.values())[0]
                model = MLP_head(n_dim, n_class=1)
            case _:
                raise ValueError(f"experiment parameter classifier type not recognized: {exp_params['classifier']}")
            
        print(f"model size: {model.get_num_params():,d}")

        # define optimizer
        if exp_params["optimizer"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=exp_params["lr"], weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=exp_params["lr"], weight_decay=0.1)

        # send model to device
        model.to(device)
        
        # train/test loop
        for epoch in tqdm.trange(exp_params["epoch"], ncols=100):
            
            # =====================   train iteration   =====================
            model.train()   # set to train mode

            for batch in train_loader.batch_iterator(exp_params["bsize"], sample_weight=exp_params["class_weights"]):
                # stop if StopIteration returned
                if batch is StopIteration:
                    break

                x, y, cw = batch

                if not(cw is None):
                    cw = cw.to(device=device)

                # compute batch loss
                pred = F.sigmoid(model(send_to_device(x, device))).view(-1)
                y = y.view(*pred.shape).to(device=device, dtype=torch.float32)
                opt.zero_grad()
                loss = F.binary_cross_entropy(pred, y, weight=cw)
                if exp_params["mean_reg_lambda"] > 0. or exp_params["variance_reg_lambda"] > 0.:   # regularization loss
                    # sample positive and nagtaives probability distributions from model
                    x_pos, x_neg = train_loader.get_random_batch_posneg(exp_params["bsize"]//2)
                    x_pos = F.sigmoid(model(send_to_device(x_pos, device)))
                    x_neg = F.sigmoid(model(send_to_device(x_neg, device)))

                    # mean divergence regularization loss
                    mean_loss = (0.5 - torch.mean(torch.cat((x_pos, x_neg), dim=0)))**2
                    
                    # variance divergence regularization loss
                    std_pos = x_pos.std()
                    std_neg = x_neg.std()
                    variance_loss = dkl_reg_loss(std_pos, std_neg) + dkl_reg_loss(std_neg, std_pos)

                    # add to classification loss
                    loss += 0.5 * (exp_params["mean_reg_lambda"] * mean_loss + exp_params["variance_reg_lambda"] * variance_loss)

                # param update
                loss.backward()
                opt.step()
            # =========================================================================

            # =====================   validation/test iteration   =====================
            model.eval() # set to eval mode

            # train/val
            for split, loader in [("train", train_loader), ("valid", valid_loader)]:
                metrics = eval(model, loader, exp_params["bsize"], device)
                for m, v in metrics.items():
                    train_metrics.append({"split": split, "metric": m, "value": v, "kfold": k, "step": epoch})
            
            # test
            metrics = eval(model, test_loader, exp_params["bsize"], device)
            for m, v in metrics.items():
                test_metrics.append({"split": "test", "metric": m, "value": v, "kfold": k, "step": epoch})

            # save checkpoint if current validation loss is lowest
            validation_loss = pandas.DataFrame(train_metrics)
            validation_loss = validation_loss[(validation_loss["metric"] == "log_loss") & (validation_loss["split"] == "valid") & (validation_loss["kfold"] == k)]
            validation_loss = validation_loss["value"].values
            if validation_loss[-1] == min(validation_loss):
                best_state_dict.update({k: model.state_dict()})
            # =========================================================================
        
    return train_metrics, test_metrics, best_state_dict, normalizer, patients_idxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["artix", "headneckctatlas", "headneckpetct", "hecktor", "oropharyngealradiomicsoutcomes", "qinheadneck", "radcure"], 
                        help='name of dataset')
    parser.add_argument('--kfold', type=int, default=5, help='number of kfold cross-validation folds')
    parser.add_argument('--train_split', type=float, default=0.8, help='proportion of training data used for training (rest for validation)')
    parser.add_argument('--epoch', type=int, default=100, help="number of training epochs")
    parser.add_argument('--modality', type=str, required=True, choices=["both", "image", "clinical"],  help="which modality to use")
    
    parser.add_argument('--task', type=str, required=True, choices=[recurrence_2y_name, recurrence_5y_name], help="task to train model on")
    parser.add_argument('--output', type=str, required=True, help='path to save results')    
    parser.add_argument('--extractors', type=str, default="ct-fm", help="extractors separated by comma (e.g., 'ct-fm,suprem')")
    parser.add_argument('--classifier', type=str, required=True, choices=["attention", "concat", "linear", "ffn"], help="classifier type")
    parser.add_argument('--n_dim', type=int, default=128, help="dimension after features fusion/concatenation")
    parser.add_argument('--n_layer', type=int, default=1, help="number of layers in attention fusion")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout probability")
    parser.add_argument('--lambda_', type=float, default=0.5, help="lambda parameter for attention trade-off")
    parser.add_argument('--mean_reg_lambda', type=float, default=0., help="lambda parameter for mean divergence regularization loss")
    parser.add_argument('--variance_reg_lambda', type=float, default=0., help="lambda parameter for variance divergence regularization loss")
    parser.add_argument('--class_weights', action='store_true', help="whether to use class weighting in the loss to counter class imbalance")
    parser.add_argument('--undersampling', action='store_true', help="whether to apply data undersampling to the training data")
    parser.add_argument('--per_center', action='store_true', help="whether to split data per center in kfold")
    parser.add_argument('--normalizer', type=str, default="scale", help="normalizer to pre-process data before training model")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--uniform_sampling', action='store_true', help="sample uniformly across centers for training")
    parser.add_argument('--gpu', type=str, default="", help='GPUs to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(device)

    out_path = pathlib.Path(args.output).joinpath(coolname.generate_slug())
    out_path.mkdir(parents=True, exist_ok=True)
    
    # experiment parameters
    exp_params = vars(args)
    exp_params["extractors"] = args.extractors.split(",")
    
    # construct data loader
    data_loader = DataLoader(base_path="./", name=exp_params["dataset"])

    # data processing
    print("loading cohorts...")
    data_loader.build_dataset()
    print("preparing data...")
    data_loader.prepare_data(exp_params, args.task)

    # save feature names
    feature_names = data_loader.X.columns.to_list()

    # fit model
    print("fitting model")
    train_metrics, test_metrics, best_state_dict, normalizer, patients_idxes = kfold_training(exp_params, data_loader, args.kfold, device,)
    
    # save results
    pandas.DataFrame(train_metrics).to_csv(out_path.joinpath("train_metrics.csv"))
    pandas.DataFrame(test_metrics).to_csv(out_path.joinpath("test_metrics.csv"))

    # save best model state dicts
    for i, state_dict in best_state_dict.items():
        torch.save(state_dict, out_path.joinpath(f"best_checkpoint_{i}.pt"))

    # save normalizers
    for i, norm in normalizer.items():
        with open(out_path.joinpath(f"normalizer_{i}.pickle"), "wb") as f:
            pickle.dump(norm, f)

    # save exp params
    exp_params["feature_names"] = feature_names
    with open(out_path.joinpath("params.json"), "w") as f:
        json.dump(exp_params, f)

    # save patients indices
    with open(out_path.joinpath("patients_idx.json"), "w") as f:
        json.dump(patients_idxes, f)

    print("done.")
