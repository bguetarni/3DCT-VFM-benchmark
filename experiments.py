import json
import yaml
import pickle
import argparse
import pathlib
import os
import pandas
import torch
import coolname

from classifiers import Attention, Concat, GatedModality, FFN, CoxModel, Classifier
from dataloader import DataLoader, CoxLoader, ProtoNetLoader
from trainer import CoxTrainer, FineTuneTrainer, ProtoNetTrainer

def display_split_stats(loader):
    y = list(loader.Y["label"].values)
    stats = {int(j): y.count(j) for j in set(y)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(y))}%)")

def kfold_training(exp_params, data_loader, kfold, device="cpu"):
    """
    Perform kfold training and testing and return classification metrics as dict

    Args:
        exp_params (dict) experiment parameters
        data_loader (DataLoader) data loader
        device (torch.device) device for model
        kfold (int) number of cross-validation folds
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

        if args.pretraining:
            print(f"pre-training model using {args.pretraining}...")

            with open("./PRETRAINING_CONFIG.yaml", "r") as f:
                params = yaml.safe_load(f)

            match args.pretraining:
                case "cox":
                    # normalize Cox pre-training features
                    cox_X = data_loader.Cox_X.copy()
                    if "image" in cox_X.columns.get_level_values("modality"):
                        cox_X.loc[:, ["image"]] = norm.transform(cox_X.loc[:, ["image"]])
                    if "age" in cox_X.columns.get_level_values("modality"):
                        cox_X.loc[:, ("clinical", "age")] = cox_X.loc[:, ("clinical", "age")].values / 100.
                    if "dose" in cox_X.columns.get_level_values("modality"):
                        cox_X.loc[:, ("clinical", "dose")] = cox_X.loc[:, ("clinical", "dose")].values / 70.
                    
                    # supplementary data for Cox pre-training provided
                    X = pandas.concat([train_loader.X.copy(), cox_X], axis=0)
                    Y = pandas.concat([train_loader.Y, data_loader.Cox_Y.copy()], axis=0)

                    # dataloader
                    loader = CoxLoader(X, Y, strategy=exp_params["cox_strategy"])
                    loader.prepare_data(exp_params["task"])

                    # model : backbone + linear layer
                    pretrain_model = CoxModel(backbone)
                    pretrain_model.to(device)
                    pretrain_model.train()

                    trainer = CoxTrainer(exp_params["cox_strategy"], params["epoch"], params["batch_size"], params["optimizer"])
                    loss = trainer.train(pretrain_model, device, loader)
                    train_metrics.extend([{"fold": k, "split": "train", "metric": "cox_loss", "value": l, "step": s} for s, l in enumerate(loss)])

                    # send backbone to cpu
                    pretrain_model.to(device="cpu")
                case "protonet":
                    # dataloader
                    loader = ProtoNetLoader(train_loader.X.copy(), train_loader.Y.copy())
                    loader.prepare_data(exp_params["task"])

                    # model : backbone + linear layer
                    backbone.to(device)
                    backbone.train()
                    
                    trainer = ProtoNetTrainer(params["n_iter"], params["batch_size"], params["optimizer"])
                    loss = trainer.train(backbone, device, loader)
                    train_metrics.extend([{"fold": k, "split": "train", "metric": "proto_loss", "value": l, "step": s} for s, l in enumerate(loss)])


                    # send backbone to cpu
                    backbone.to(device="cpu")
                case _:
                    raise ValueError(f"pre-training task {args.pretraining} not implemented, see pretraining argument choices")
                
        # build classifier model from backbone
        binaryclassifier = Classifier(backbone, freeze_backbone=exp_params["freeze_backbone"])

        # train model
        optimizer_params = {"name": exp_params["optimizer"], "lr": exp_params["lr"]}
        trainer = FineTuneTrainer(exp_params["epoch"], exp_params["bsize"], optimizer_params, bool(exp_params["lr_scheduler"]), exp_params["class_weights"])
        fold_train_metrics, fold_test_metrics, fold_best_state_dict = trainer.train(binaryclassifier, device, train_loader, valid_loader, test_loader)

        # update metrics and checkpoint
        best_state_dict.update({k: fold_best_state_dict})
        train_metrics.extend([{"fold": k, **d} for d in fold_train_metrics])
        test_metrics.extend([{"fold": k, **d} for d in fold_test_metrics])
        
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
    parser.add_argument('--pretraining', default=None, choices=["cox", "protonet"], help="which method to use for pre-training")
    parser.add_argument('--cox_strategy', default="1v1", choices=["1v1", "1vN"], help="which Cox pre-training strategy to use: 1v1 (train on pairs of one positive and one negative samples) or 1vN (train on batches with multiple positive and negative samples using Cox partial likelihood loss)")
    parser.add_argument('--modality', type=str, required=True, choices=["both", "image", "clinical"],  help="which modality to use")
    parser.add_argument('--task', type=str, required=True, choices=["rfs_2", "rfs_5"], help="task to train model on")
    parser.add_argument('--output', type=str, required=True, help='path to save results')    
    parser.add_argument('--extractors', type=str, default="ct-fm", help="extractors separated by comma (e.g., 'ct-fm,suprem')")
    parser.add_argument('--backbone', type=str, required=True, choices=["attention", "concat", "gated", "ffn"], help="type of backbone")
    parser.add_argument('--freeze_backbone', action='store_true', help="freeze the backbone for binary task fine-tuning")
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
    
    print("Script arguments:")
    for k, v in vars(args).items():
        print(f"    {k}: {v}")
    
    main(args)

    print("done.")
