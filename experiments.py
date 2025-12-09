import json
import pickle
import argparse
import pathlib
import os
from itertools import chain
import pandas
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
import torch
import torch.nn.functional as F
import coolname

from dataloader import ARTIX, HECKTOR, HeadNeckCTAtlas, HeadNeckPETCT, QINHEADNECK, OropharyngealRadiomicsOutcomes, RADCURE
from dataloader import recurrence_2y_name, recurrence_5y_name
from classifiers.classifiers import Attention, Linear, Concat, Normalizer


class DataLoader:
    def __init__(self, base_path=None, X=None, Y=None, uniform_sampling=False):
        self.base_path = base_path
        self.X = X
        self.Y = Y
        self.uniform_sampling = uniform_sampling

    def len(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X.loc[idx], self.Y.loc[idx]

    def build_dataset(self):
        features = pandas.DataFrame([])
        labels = pandas.DataFrame([])
        for loader in tqdm.tqdm([ARTIX, HECKTOR, HeadNeckCTAtlas, HeadNeckPETCT, QINHEADNECK, OropharyngealRadiomicsOutcomes, RADCURE], ncols=100):
            loader = loader(None)
            x, y = loader.build_dataset(self.base_path)
            features = pandas.concat((features, x))
            labels = pandas.concat((labels, y))
        self.X = features
        self.Y = labels

    def prepare_data(self, exp_params, task):
        features_name = list(chain.from_iterable(i.keys() for i in exp_params["dims"].values()))
        self.X = self.X[self.X["features"].isin(features_name)]
        self.X = self.X.pivot(index="patient", columns=["modality", "features", "name"], values="value")
        self.X = self.X.sort_index(axis="columns")

        self.Y = self.Y.set_index("patient")
        if not(task in self.Y.columns):
            raise ValueError(f"task {task} not valid. Validones are [{recurrence_2y_name}, {recurrence_5y_name}]")
        self.Y = self.Y[["center", task]].rename(columns={task: "label"})

        invalid_label_patients = self.Y[pandas.isna(self.Y["label"])].index
        self.X = self.X.drop(index=invalid_label_patients, errors="ignore")
        self.Y = self.Y.drop(index=invalid_label_patients)

    def split(self, train_centers, test_centers, train_val_factor=0.8):
        assert set(train_centers).isdisjoint(set(test_centers)), "train and test centers must be disjoint"

        if train_val_factor is None:
            X_train = self.X
            Y_train = self.Y
            X_valid = None
            Y_valid = None
        else:
            train_idx = list(self.Y["center"].isin(train_centers).index)
            np.random.shuffle(train_idx)
            n = int(train_val_factor*len(train_idx))
            val_idx = train_idx[n:]
            train_idx = train_idx[:n]

            X_train = self.X.loc[self.X.index.isin(train_idx)]
            Y_train = self.Y.loc[X_train.index]

            X_valid = self.X.loc[self.X.index.isin(val_idx)]
            Y_valid = self.Y.loc[X_valid.index]

        Y_test = self.Y[self.Y["center"].isin(test_centers)]
        X_test = self.X.loc[Y_test.index]
        
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    
    def shuffle(self):
        idx = np.random.permutation(self.X.index)
        self.X = self.X.reindex(idx)
        self.Y = self.Y.reindex(idx)

    def get_random_batch(self, n):
        """
        get random batch taking into account centers equality
        """
        if self.uniform_sampling:
            centers = self.Y["center"].unique()
            n_per_center = int(np.ceil(n / len(centers)))
            idx = [np.random.choice(self.Y[self.Y["center"] == c].index, size=n_per_center, replace=False) for c in centers]
            idx = list(chain.from_iterable(idx))[:n]
        else:
            idx = np.random.choice(self.X.index, size=n, replace=False)
        
        x = self.X.loc[idx]
        y = self.Y.loc[idx]

        x = {m: {
                k: torch.tensor(x[(m, k)].values, dtype=torch.float32) for k in x.columns.get_level_values("features")[x.columns.get_level_values("modality") == m].unique()
            } for m in x.columns.get_level_values("modality").unique()}
        y = torch.tensor(y["label"].values)

        return x, y
    
    def get_random_batch_posneg(self, n):
        if self.uniform_sampling:
            centers = self.Y["center"].unique()
            n_per_center = int(np.ceil(n / len(centers)))
            
            x_pos_idx = [np.random.choice(self.Y[(self.Y["center"] == c) & (self.Y["label"] == 1)].index, size=n_per_center) for c in centers]
            x_pos = self.X.loc[list(chain.from_iterable(x_pos_idx))[:n]]
            
            x_neg_idx = [np.random.choice(self.Y[(self.Y["center"] == c) & (self.Y["label"] == 0)].index, size=n_per_center) for c in centers]
            x_neg = self.X.loc[list(chain.from_iterable(x_neg_idx))[:n]]
        else:
            x_pos = self.X.loc[np.random.choice(self.Y[self.Y["label"] == 1].index, size=n)]
            x_neg = self.X.loc[np.random.choice(self.Y[self.Y["label"] == 0].index, size=n)]

        x_pos = {m: {
                k: torch.tensor(x_pos[(m, k)].values, dtype=torch.float32) for k in x_pos.columns.get_level_values("features")[x_pos.columns.get_level_values("modality") == m].unique()
            } for m in x_pos.columns.get_level_values("modality").unique()}
        
        x_neg = {m: {
                k: torch.tensor(x_neg[(m, k)].values, dtype=torch.float32) for k in x_neg.columns.get_level_values("features")[x_neg.columns.get_level_values("modality") == m].unique()
            } for m in x_neg.columns.get_level_values("modality").unique()}
        
        return x_pos, x_neg
    
    def batch_iterator(self, n):
        for idx in range(0, len(self.Y), n):   # handle case where dataset smaller than batch size
            indices = self.Y.index[idx:idx+n]
            x = self.X.loc[indices]
            y = self.Y.loc[indices]

            x = {m: {
                k: torch.tensor(x[(m, k)].values, dtype=torch.float32) for k in x.columns.get_level_values("features")[x.columns.get_level_values("modality") == m].unique()
            } for m in x.columns.get_level_values("modality").unique()}
            y = torch.tensor(y["label"].values)

            yield x, y
        return StopIteration

    def split_per_center(self):
        dataloader = []
        for center in self.Y["center"].unique():
            y = self.Y[self.Y["center"] == center]
            x = self.X.loc[y.index]
            dataloader.append((center, DataLoader(self.base_path, x, y, self.uniform_sampling)))
        return dataloader


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
            "auc": roc_auc_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "specificity": zero_division(tn, (tn + fp)),
            "sensitivity": zero_division(tp, (tp + fn)),
            "log_loss": log_loss(y, y_pred_proba, labels=[0,1])}
    return m

def eval(model, loader, batch_size, device, per_center=False):
    if per_center:
        metrics = {}
        for center, center_loader in loader.split_per_center():
            metrics.update({center: eval(model, center_loader, batch_size, device)})
        return metrics
    else:
        y_pred_proba = []
        y_pred = []
        y_true = []
        for batch in loader.batch_iterator(batch_size):
            if batch == StopIteration:
                break
            x, y = batch
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

def cross_validation(exp_params, data_loader, device="cpu", bootstrap=1):
    """
    Perform bootstrap training and testing and return classification metrics as dict

    Args:
        exp_params (dict) experiment parameters
        data_loader (DataLoader) data loader
        device (torch.device) device for model
        bootstrap (int) number of bootstraps
    """

    train_metrics = []
    test_metrics = []
    best_state_dict = {}
    normalizer = {}
    for b in range(bootstrap):
        print(f"bootstrap {b+1}/{bootstrap}")

        # split into train and test based on centers
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = data_loader.split(INTERNAL_CENTERS, EXTERNAL_CENTERS)

        # apply normalization function column-wise
        print("normalizing features values..")
        norm = Normalizer(args.normalizer)
        X_train = norm.fit_transform(X_train)
        X_valid = norm.transform(X_valid)
        X_test = norm.transform(X_test)
        normalizer.update({b: norm.get_params()})

        train_loader = DataLoader(X=X_train, Y=Y_train, uniform_sampling=exp_params["uniform_sampling"])
        valid_loader = DataLoader(X=X_valid, Y=Y_valid)
        test_loader = DataLoader(X=X_test, Y=Y_test)

        print("training data stat:")
        print(display_split_stats(train_loader))
        print("validation data stat:")
        print(display_split_stats(valid_loader))
        print("testing data stat:")
        print(display_split_stats(test_loader))

        match exp_params["classifier"]:
            case "attention":
                model = Attention(**exp_params)
            case "linear":
                n_dim = list(chain.from_iterable([v.values() for v in exp_params["dims"].values()]))[0]
                model = Linear(n_dim, n_class=1)
            case "concat":
                model = Concat(**exp_params)
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
        for n_iter in tqdm.trange(exp_params["n_iter"], ncols=100):
            # eval iteration (on both train and test data)
            if (n_iter + 1) % exp_params["eval_iter"] == 0:
                model.eval()

                # train/val
                for split, loader in [("train", train_loader), ("valid", valid_loader)]:
                    metrics = eval(model, loader, exp_params["bsize"], device)
                    for m, v in metrics.items():
                        train_metrics.append({"split": split, "metric": m, "value": v, "bootstrap": b, "step": n_iter})
                
                # test
                metrics = eval(model, test_loader, exp_params["bsize"], device, per_center=True)
                for center, ms in metrics.items():
                    for m, v in ms.items():
                        test_metrics.append({"center": center, "split": "test", "metric": m, "value": v, "bootstrap": b, "step": n_iter})

                # save checkpoint if current validation loss is lowest
                validation_loss = pandas.DataFrame(train_metrics)
                validation_loss = validation_loss[(validation_loss["metric"] == "log_loss") & (validation_loss["split"] == "valid") & (validation_loss["bootstrap"] == b)]
                validation_loss = validation_loss["value"].values
                if validation_loss[-1] == min(validation_loss):
                    best_state_dict.update({b: model.state_dict()})
            
            # train iteration
            model.train()   # set to train mode
            x, y = train_loader.get_random_batch(exp_params["bsize"])
            pred = model(send_to_device(x, device))
            y = y.view(*pred.shape).to(device=device, dtype=torch.float32)
            opt.zero_grad()
            loss = F.binary_cross_entropy(F.sigmoid(pred), y)
            if exp_params["mean_reg_lambda"] > 0. or exp_params["variance_reg_lambda"] > 0.:   # regularization loss
                x_pos, x_neg = train_loader.get_random_batch_posneg(exp_params["bsize"]//2)
                x_pos = model(send_to_device(x_pos, device))
                x_neg = model(send_to_device(x_neg, device))

                # mean divergence regularization loss
                mean_loss = (0.5 - torch.mean(torch.cat((x_pos, x_neg), dim=0)))**2
                
                # variance divergence regularization loss
                std_pos = x_pos.std()
                std_neg = x_neg.std()
                variance_loss = dkl_reg_loss(std_pos, std_neg) + dkl_reg_loss(std_neg, std_pos)

                # add to classification loss
                loss += 0.5 * (exp_params["mean_reg_lambda"] * mean_loss + exp_params["variance_reg_lambda"] * variance_loss)
            loss.backward()
            opt.step()
        
    return train_metrics, test_metrics, best_state_dict, normalizer


INTERNAL_CENTERS = ["CHUM", "CHUP", "CHUS", "HGJ", "HMR", "MDA", "USZ", "UHN"]
EXTERNAL_CENTERS = ["CEM", "UIHC"]

DIMS = {
    "image": {"ct-fm": 512, "suprem": 128, "vista3d": 768}, 
    "clinical": {"llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb": 768, "llm-BioLORD-2023-M": 768, "llm-embeddinggemma-300m-medical": 768},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=[recurrence_2y_name, recurrence_5y_name], help="task to train model on")
    parser.add_argument('--output', type=str, required=True, help='path to save results')
    parser.add_argument('--bootstrap', type=int, default=1, help='number of bootstraps')
    parser.add_argument('--extractors', type=str, required=True, help="extractors separated by comma (e.g., 'ct-fm,suprem')")
    parser.add_argument('--classifier', type=str, required=True, choices=["attention", "concat", "linear"], help="classifier type")
    parser.add_argument('--n_dim', type=int, default=128, help="dimension after features fusion/concatenation")
    parser.add_argument('--n_layer', type=int, default=1, help="number of layers in attention fusion")
    parser.add_argument('--lambda_', type=float, default=0.5, help="lambda parameter for attention trade-off")
    parser.add_argument('--mean_reg_lambda', type=float, default=0., help="lambda parameter for mean divergence regularization loss")
    parser.add_argument('--variance_reg_lambda', type=float, default=0., help="lambda parameter for variance divergence regularization loss")
    parser.add_argument('--normalizer', type=str, default="scale", help="normalizer to pre-process data before training model")
    parser.add_argument('--pca', type=int, default=None, help="dimension after applying PCA, set to None to not apply")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=1000, help="number of training iterations")
    parser.add_argument('--eval_iter', type=int, default=10, help="number of training iterations between evaluations")
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
    extractors = args.extractors.split(",")
    exp_params.update({"dims": {m: {k: v for k,v in DIMS[m].items() if k in extractors} for m in DIMS.keys()}})
    
    # construct data loader
    data_loader = DataLoader(base_path="./")

    # data processing
    print("loading cohorts...")
    data_loader.build_dataset()
    print("preparing data...")
    data_loader.prepare_data(exp_params, args.task)

    # fit model
    print("fitting model")
    train_metrics, test_metrics, best_state_dict, normalizer = cross_validation(exp_params, data_loader, device, args.bootstrap)
    
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
    with open(out_path.joinpath("params.json"), "w") as f:
        json.dump(exp_params, f)

    print("done.")
