import json
import yaml
import pickle
import argparse
import pathlib
import os
import pandas
import torch
import coolname

from classifiers import Attention, Concat, GatedModality, FFN, Classifier
from dataloader import DataLoader, CoxProtoNetLoader, ProtoNetLoader
from trainer import CoxProtoNetTrainer, FineTuneTrainer, ProtoNetTrainer

def display_split_stats(loader):
    y = list(loader.Y["label"].values)
    stats = {int(j): y.count(j) for j in set(y)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(y))}%)")

def bootstrap_kfold_training(exp_params, data_loader, device="cpu"):
    """
    Perform bootstrap/CV training and testing and return classification metrics as dict

    Args:
        exp_params (dict) experiment parameters
        data_loader (DataLoader) data loader
        device (torch.device) device for model
    """

    if exp_params["undersampling"]:
        print("applying undersampling")
        data_loader.undersampling(per_center=args.per_center)

    metrics = []
    best_state_dict = {}
    normalizer = {}
    patients_idxes = {}
    test_pred_proba = {}

    if exp_params['kfold']:
        iter_ = data_loader.split(kfold=exp_params['kfold'])
    else:
        iter_ = data_loader.split(bootstrap=exp_params['bootstrap'])

    for i, ((X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)) in enumerate(iter_):
        if exp_params['kfold']:
            print(f"run {i+1}/{exp_params['kfold']}")
        else:
            print(f"run {i+1}/{exp_params['bootstrap']}")
        train_loader = DataLoader(base_path=None, name=None, X=X_train, Y=Y_train, uniform_sampling=exp_params["uniform_sampling"], class_weights=exp_params["class_weights"])
        valid_loader = DataLoader(base_path=None, name=None, X=X_valid, Y=Y_valid)
        test_loader = DataLoader(base_path=None, name=None, X=X_test, Y=Y_test)

        # save patients indices for current fold
        patients_idxes.update({f"run {i}": {
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
            normalizer.update({i: norm.get_params()})

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

        if args.pretraining:
            print(f"pre-training model using {args.pretraining}...")

            with open("./PRETRAINING_CONFIG.yaml", "r") as f:
                params = yaml.safe_load(f)

            match args.pretraining:
                case "cox+protonet":
                    loader = CoxProtoNetLoader(train_loader.X.copy(), train_loader.Y.copy())
                    trainer = CoxProtoNetTrainer(params["n_iter"], params["batch_size"], params["optimizer"])
                    loss_type = "cox_protonet_loss"
                case "protonet":
                    loader = ProtoNetLoader(train_loader.X.copy(), train_loader.Y.copy())                    
                    trainer = ProtoNetTrainer(params["n_iter"], params["batch_size"], params["optimizer"])
                    loss_type = "proto_loss"
                case _:
                    raise ValueError(f"pre-training task {args.pretraining} not implemented, see pretraining argument choices")
                
            # pre-train the backbone
            loader.prepare_data(exp_params["task"])
            backbone.to(device)            
            loss = trainer.train(backbone, device, loader)
            metrics.extend([{"run": i, "split": "train", "metric": loss_type, "value": l, "step": s} for s, l in enumerate(loss)])

            # send backbone back to cpu to build classifier
            backbone.to(device="cpu")
                
        # build classifier model from backbone
        model = Classifier(backbone)

        # train model
        try:
            optimizer_params = {"name": exp_params["optimizer"], "lr": exp_params["lr"]}
            trainer = FineTuneTrainer(exp_params["epoch"], exp_params["bsize"], optimizer_params, bool(exp_params["lr_scheduler"]), exp_params["class_weights"])
            fold_metrics, fold_best_state_dict, fold_test_pred_proba = trainer.train(model, device, train_loader, valid_loader, test_loader)
        except RuntimeError:
            print("RuntimeError during training, skipping run")
            continue

        # update metrics and checkpoint
        best_state_dict.update({i: fold_best_state_dict})
        test_pred_proba.update({i: fold_test_pred_proba})
        metrics.extend([{"run": i, **d} for d in fold_metrics])
        
    return metrics, best_state_dict, normalizer, patients_idxes, test_pred_proba


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
    metrics, best_state_dict, normalizer, patients_idxes, test_pred_proba = bootstrap_kfold_training(exp_params, data_loader, device)
    
    # save results
    pandas.DataFrame(metrics).to_csv(out_path.joinpath("metrics.csv"))

    # save best model state dicts
    torch.save(best_state_dict, out_path.joinpath(f"best_checkpoint.pt"))

    # save normalizers
    with open(out_path.joinpath(f"normalizer.pickle"), "wb") as f:
        pickle.dump(normalizer, f)

    # save exp params
    with open(out_path.joinpath("params.json"), "w") as f:
        json.dump(exp_params, f)

    # save patients indices
    with open(out_path.joinpath("patients_idx.json"), "w") as f:
        json.dump(patients_idxes, f)

    # save feature names
    with open(out_path.joinpath("features.json"), "w") as f:
        json.dump(feature_names, f)

    # save test predicted probabilities and true labels for each bootstrap iteration
    with open(out_path.joinpath("test_pred_proba.json"), "w") as f:
        json.dump(test_pred_proba, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=["artix", "headneckctatlas", "headneckpetct", "hecktor", "oropharyngealradiomicsoutcomes", "qinheadneck", "radcure"], 
                        help='name of dataset')
    parser.add_argument('--kfold', type=int, default=None, help='number of folds for k-fold cross-validation')
    parser.add_argument('--bootstrap', type=int, default=None, help='number of bootstrap training')
    parser.add_argument('--pretraining', default=None, choices=["cox+protonet", "protonet"], help="which method to use for pre-training")
    parser.add_argument('--modality', type=str, required=True, choices=["both", "image", "clinical"],  help="which modality to use")
    parser.add_argument('--task', type=str, required=True, choices=["rfs_2", "rfs_5"], help="task to train model on")
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', action='store_true', help="apply scheduling to lerning rate")
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--mean_reg_lambda', type=float, default=0., help="lambda parameter for mean divergence regularization loss")
    parser.add_argument('--variance_reg_lambda', type=float, default=0., help="lambda parameter for variance divergence regularization loss")
    parser.add_argument('--class_weights', action='store_true', help="whether to use class weighting in the loss to counter class imbalance")
    parser.add_argument('--undersampling', action='store_true', help="whether to apply data undersampling to the training data")
    parser.add_argument('--per_center', action='store_true', help="whether to split data per center in kfold")
    parser.add_argument('--uniform_sampling', action='store_true', help="sample uniformly across centers for training")
    parser.add_argument('--gpu', type=str, default="", help='GPUs to use')
    args = parser.parse_args()

    assert (args.kfold is not None) ^ (args.bootstrap is not None), "exactly one of kfold or bootstrap argument must be provided"
    
    print("Script arguments:")
    for k, v in vars(args).items():
        print(f"    {k}: {v}")
    
    main(args)

    print("done.")
