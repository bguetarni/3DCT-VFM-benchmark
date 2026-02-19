import json
import yaml
import pickle
import argparse
import pathlib
import os
import pandas
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
import torch
import torch.nn.functional as F
import coolname

from classifiers import Attention, Concat, GatedModality, FFN, CoxModel, Classifier
from dataloader import DataLoader, CoxLoader


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

def cox_pretrain(backbone, X, Y, exp_params, device, epsilon=1e-5, **kwargs):
    if "cox_X" in kwargs.keys():
        # supplementary data for Cox pre-training provided
        X = pandas.concat([X, kwargs["cox_X"]], axis=0)
        Y = pandas.concat([Y, kwargs["cox_Y"]], axis=0)

    # dataloader
    loader = CoxLoader(X, Y, strategy=exp_params["cox_strategy"])
    loader.prepare_data(exp_params["task"])

    # model : backbone + linear layer
    model = CoxModel(backbone, exp_params["cox_strategy"])
    model.to(device)
    model.train()

    with open("./COX_PRETRAINING_CONFIG.yaml", "r") as f:
        params = yaml.safe_load(f)

    if params["optimizer"]["name"] == "adam":
        opt = torch.optim.Adam(model.parameters(), weight_decay=0.1)
    else:
        opt = torch.optim.SGD(model.parameters(), weight_decay=0.1)
    
    # create learning rate scheduler (cosine annealing with warmup)
    total_steps = int(params["epoch"] * np.ceil(len(loader) / params["batch_size"]))
    warmup = params["optimizer"]["warmup"] / params["epoch"]
    div_factor = float(params["optimizer"]["max_lr"]) / float(params["optimizer"]["initial_lr"])
    final_div_factor = float(params["optimizer"]["initial_lr"]) / float(params["optimizer"]["final_lr"])
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                     max_lr=float(params["optimizer"]["max_lr"]),
                                                     total_steps=total_steps, 
                                                     pct_start=warmup, 
                                                     anneal_strategy='cos',
                                                     div_factor=div_factor,
                                                     final_div_factor=final_div_factor)
    
    for _ in tqdm.trange(params["epoch"], ncols=100):
        for batch in loader.batch_iterator(params["batch_size"]):
            if batch is StopIteration:
                break
            neg, pos = batch
            if exp_params["cox_strategy"] == "1v1":
                neg = model(send_to_device(neg, device))
                pos = model(send_to_device(pos, device))
                cox_loss = torch.mean(-torch.log(torch.exp(neg) / (torch.exp(pos) + epsilon)))
            else:
                neg = model(send_to_device(neg, device))
                fn = lambda t: model(send_to_device(t, device))
                pos = list(map(fn, pos))
                pos = torch.stack(pos, dim=1)
                cox_loss = torch.mean(-torch.log(torch.exp(neg) / (torch.exp(pos).sum(dim=1) + epsilon)))
            opt.zero_grad()
            cox_loss.backward()
            opt.step()
            scheduler.step()
    
    # send back the backbone to cpu device
    model.to(device="cpu")

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
    for k, ((X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)) in enumerate(data_loader.split(kfold, 
                                                                                                     train_val_split=args.train_split, 
                                                                                                     per_center=args.per_center)):
        print(f"kfold {k+1}/{kfold}")
        train_loader = DataLoader(base_path=None, name=None, X=X_train, Y=Y_train, uniform_sampling=exp_params["uniform_sampling"], class_weights=exp_params["class_weights"])
        valid_loader = DataLoader(base_path=None, name=None, X=X_valid, Y=Y_valid)
        test_loader = DataLoader(base_path=None, name=None, X=X_test, Y=Y_test)

        # save patients indices for current fold
        patients_idxes.update({f"fold {k}": {
            "train": train_loader.X.index.to_list(), 
            "valid": valid_loader.X.index.to_list(), 
            "test": test_loader.X.index.to_list()}})

        # construct dict of number of dimensions for each modality and features
        columns_names = train_loader.X.columns
        dims = {}
        dims.update({"clinical": len([c for c in columns_names if c[0] == "clinical"])})
        dims.update({"image": {}})
        for f in set([c[1] for c in columns_names if c[0] != "clinical"]):
            dims["image"].update({f: len([c for c in columns_names if c[0] == "image" and c[1] == f])})
        exp_params.update({"dims": dims})

        # apply normalization function column-wise to image features
        print("normalizing features values..")
        if "image" in train_loader.X.columns.get_level_values("modality"):
            norm = train_loader.normalize_image_features(normalizer=args.normalizer)
            valid_loader.normalize_image_features(normalizer=norm)
            test_loader.normalize_image_features(normalizer=norm)
            normalizer.update({k: norm.get_params()})

        # normalize clinical features
        # only present features will be normalized
        train_loader.normalize_clinical_features()
        valid_loader.normalize_clinical_features()
        test_loader.normalize_clinical_features()

        print("training data stat:")
        print(display_split_stats(train_loader))
        print("validation data stat:")
        print(display_split_stats(valid_loader))
        print("testing data stat:")
        print(display_split_stats(test_loader))

        match exp_params["backbone"]:
            case "attention":
                backbone = Attention(**exp_params)
            case "concat":
                backbone = Concat(**exp_params)
            case "ffn":
                if len(dims["image"]) > 0:
                    in_dim = sum(dims["image"].values())
                else:
                    in_dim = dims["clinical"]
                backbone = FFN(in_dim, exp_params["hidden_dim"], exp_params["out_dim"])
            case "gated":
                backbone = GatedModality(**exp_params)
            case _:
                raise ValueError(f"experiment parameter backbone type not recognized: {exp_params['backbone']}")
            
        print(f"backbone size: {backbone.get_num_params():,d}")

        if args.cox_pretraining:
            cox_X = data_loader.Cox_X.copy()
            cox_Y = data_loader.Cox_Y.copy()
            
            # normalize Cox pre-training features
            if "image" in cox_X.columns.get_level_values("modality"):
                cox_X.loc[:, ["image"]] = norm.transform(cox_X.loc[:, ["image"]])
            if "age" in cox_X.columns.get_level_values("modality"):
                cox_X.loc[:, ("clinical", "age")] = cox_X.loc[:, ("clinical", "age")].values / 100.
            if "dose" in cox_X.columns.get_level_values("modality"):
                cox_X.loc[:, ("clinical", "dose")] = cox_X.loc[:, ("clinical", "dose")].values / 70.
            
            # pre-train model on Cox task (use .copy() on DataFrame to avoid modifying original data_loader which are used for each fold)
            print("pre-training model using Cox like task...")
            cox_pretrain(backbone, train_loader.X.copy(), train_loader.Y.copy(), exp_params, device, cox_X=cox_X, cox_Y=cox_Y)

        # build classifier model from backbone
        model = Classifier(backbone)

        # define optimizer
        if exp_params["optimizer"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=exp_params["lr"], weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=exp_params["lr"], weight_decay=0.1)

        # send model to device
        model.to(device)
        
        # train/test loop
        print("training binary classifier...")
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


def main(args):
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
    data_loader.load()
    print("preprocess data...")
    data_loader.preprocess(exp_params, args.task)

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
    with open(out_path.joinpath("params.json"), "w") as f:
        json.dump(exp_params, f)

    # save patients indices
    with open(out_path.joinpath("patients_idx.json"), "w") as f:
        json.dump(patients_idxes, f)

    # save feature names
    with open(out_path.joinpath("features.json"), "w") as f:
        json.dump(feature_names, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=["artix", "headneckctatlas", "headneckpetct", "hecktor", "oropharyngealradiomicsoutcomes", "qinheadneck", "radcure"], 
                        help='name of dataset')
    parser.add_argument('--kfold', type=int, default=5, help='number of kfold cross-validation folds')
    parser.add_argument('--train_split', type=float, default=0.8, help='proportion of training data used for training (rest for validation)')
    parser.add_argument('--cox_pretraining', action='store_true', help="whether to first pre-train using Cox like task before fine-tuning on classification task")
    parser.add_argument('--cox_strategy', default="1v1", choices=["1v1", "1vN"], help="which Cox pre-training strategy to use: 1v1 (train on pairs of one positive and one negative samples) or 1vN (train on batches with multiple positive and negative samples using Cox partial likelihood loss)")
    parser.add_argument('--modality', type=str, required=True, choices=["both", "image", "clinical"],  help="which modality to use")
    parser.add_argument('--task', type=str, required=True, choices=["R2y", "R5y"], help="task to train model on")
    parser.add_argument('--output', type=str, required=True, help='path to save results')    
    parser.add_argument('--extractors', type=str, default="ct-fm", help="extractors separated by comma (e.g., 'ct-fm,suprem')")
    parser.add_argument('--backbone', type=str, required=True, choices=["attention", "concat", "gated", "ffn"], help="type of backbone")
    parser.add_argument('--hidden_dim', type=int, default=128, help="dimension after features fusion")
    parser.add_argument('--out_dim', type=int, default=64, help="output dimension of backbone")
    parser.add_argument('--n_layer', type=int, default=1, help="number of layers in attention fusion")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout probability")
    parser.add_argument('--lambda_', type=float, default=0.5, help="lambda parameter for attention trade-off")
    parser.add_argument('--normalizer', type=str, default="scale", choices=["scale", "minmax", "unit"], help="normalizer to pre-process data before training model")    
    parser.add_argument('--epoch', type=int, default=100, help="number of training epochs")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--mean_reg_lambda', type=float, default=0., help="lambda parameter for mean divergence regularization loss")
    parser.add_argument('--variance_reg_lambda', type=float, default=0., help="lambda parameter for variance divergence regularization loss")
    parser.add_argument('--class_weights', action='store_true', help="whether to use class weighting in the loss to counter class imbalance")
    parser.add_argument('--undersampling', action='store_true', help="whether to apply data undersampling to the training data")
    parser.add_argument('--per_center', action='store_true', help="whether to split data per center in kfold")
    parser.add_argument('--uniform_sampling', action='store_true', help="sample uniformly across centers for training")
    parser.add_argument('--gpu', type=str, default="", help='GPUs to use')
    args = parser.parse_args()

    main(args)

    print("done.")
