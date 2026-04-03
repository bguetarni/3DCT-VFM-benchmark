import json
import yaml
import pickle
import argparse
import pathlib
import os
import pandas
import torch
import coolname
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from classifiers import Attention, Concat, GatedModality, FFN, Classifier
from dataloader import Data, DataLoader, CoxProtoNetLoader, ProtoNetLoader
from trainer import CoxProtoNetTrainer, FineTuneTrainer, ProtoNetTrainer

def display_split_stats(loader):
    y = loader.get_labels()
    stats = {int(j): y.count(j) for j in set(y)}
    for j, c in stats.items():
        print(f"\t {j}: {c} ({int(100*c/len(y))}%)")

def bootstrap_training(exp_params, internal_loader, external_loader, device="cpu"):
    """
    Perform bootstrap training and testing and return classification metrics as dict

    Args:
        exp_params (dict) experiment parameters
        internal_loader (DataLoader) data loader for internal dataset to use for pre-training and training
        external_loader (List[DataLoader]) list of data loaders for external datasets to use for external validation
        device (torch.device,str) device for model training and inference
    """

    # split internal dataset into training and validation
    training_loader, validation_loader = internal_loader.split_loader()

    # apply normalization function to image features
    print("normalizing features values..")
    if exp_params["modality"] in ["both", "image"]:
        normalizer = training_loader.normalize_image_features(normalizer=exp_params["normalizer"])
        validation_loader.normalize_image_features(normalizer)
        for external in external_loader:
            external.normalize_image_features(normalizer)
    else:
        normalizer = None
    
    if exp_params["modality"] in ["both", "clinical"]:
        # normalize clinical features
        training_loader.normalize_clinical_features()
        validation_loader.normalize_clinical_features()
        for external in external_loader:
            external.normalize_clinical_features()
        
        # get max value for each clinical feature across datasets to know how many dimensions 
        # to use for one-hot encoding categorical features
        max_clinical_values = training_loader.get_max_clinical_values()
        for k, v in validation_loader.get_max_clinical_values().items():
            max_clinical_values[k] = max(max_clinical_values[k], v)
        for external in external_loader:
            for k, v in external.get_max_clinical_values().items():
                max_clinical_values[k] = max(max_clinical_values[k], v)
        # one-hot encode categorical clinical features using max values calculated above
        training_loader.one_hot_encode_clinical_features(max_clinical_values)
        validation_loader.one_hot_encode_clinical_features(max_clinical_values)
        for external in external_loader:
            external.one_hot_encode_clinical_features(max_clinical_values)

    # construct dict of number of dimensions for each modality and features
    dims = training_loader.get_features_dimension()
    exp_params.update({"dims": dims})

    # convert features to tensor
    training_loader.convert_to_tensor()
    validation_loader.convert_to_tensor()
    for external in external_loader:
        external.convert_to_tensor()

    metrics = []
    best_state_dict = {}
    for b in range(exp_params["bootstrap"]):
        print(f"run {b+1}/{exp_params['bootstrap']}")

        # apply undersampling to internal training split data
        # udnersampling is done at each bootstrap iteration to get different samples for the majority class
        if exp_params["undersampling"]:
            print("applying undersampling")
            bootstrap_training_loader = training_loader.undersampling()
        else:
            bootstrap_training_loader = training_loader

        print("training data stat:")
        print(display_split_stats(bootstrap_training_loader))
        print("validation data stat:")
        print(display_split_stats(validation_loader))
        print("testing data stat:")
        for test_loader in external_loader:
            print(test_loader.data.cohort)
            print(display_split_stats(test_loader))
        
        match exp_params["backbone"]:
            case "attention":
                backbone = Attention(**exp_params)
            case "concat":
                backbone = Concat(**exp_params)
            case "ffn":
                if dims["image"] and len(dims["image"]) > 0:
                    in_dim = sum(dims["image"].values())
                else:
                    in_dim = dims["clinical"]
                backbone = FFN(in_dim, exp_params["hidden_dim"], exp_params["out_dim"])
            case "gated":
                backbone = GatedModality(**exp_params)
            case _:
                raise ValueError(f"experiment parameter backbone type not recognized: {exp_params['backbone']}")
            
        print(f"backbone size: {backbone.get_num_params():,d}")

        if exp_params["pretraining"]:
            print(f"pre-training model using {exp_params['pretraining']}...")

            with open("./PRETRAINING_CONFIG.yaml", "r") as f:
                params = yaml.safe_load(f)

            match exp_params["pretraining"]:
                case "cox+protonet":
                    loader = CoxProtoNetLoader(bootstrap_training_loader.data)
                    trainer = CoxProtoNetTrainer(params["n_iter"], params["batch_size"], params["optimizer"])
                    loss_type = "cox_protonet_loss"
                case "protonet":
                    loader = ProtoNetLoader(bootstrap_training_loader.data)
                    trainer = ProtoNetTrainer(params["n_iter"], params["batch_size"], params["optimizer"])
                    loss_type = "proto_loss"
                case _:
                    raise ValueError(f"pre-training task {exp_params['pretraining']} not implemented, see pretraining argument choices")
                
            # pre-train the backbone
            loader.prepare_data(exp_params["task"])
            backbone.to(device)            
            loss = trainer.train(backbone, device, loader)
            metrics.extend([{"run": b, "split": "train", "metric": loss_type, "value": l, "step": s} for s, l in enumerate(loss)])

            # send backbone back to cpu to build classifier
            backbone.to(device="cpu")
                
        # build classifier model from backbone
        freeze_ = (exp_params["pretraining"] and exp_params["freeze"])
        model = Classifier(backbone, freeze=freeze_)

        # train model
        optimizer_params = {"name": exp_params["optimizer"], "lr": exp_params["lr"]}
        trainer = FineTuneTrainer(exp_params["epoch"], exp_params["bsize"], optimizer_params, bool(exp_params["lr_scheduler"]), exp_params["class_weights"])
        fold_metrics, fold_best_state_dict = trainer.train(model, device, bootstrap_training_loader, validation_loader, external_loader, **exp_params)

        # update metrics and checkpoint
        best_state_dict.update({b: fold_best_state_dict})
        metrics.extend([{"run": b, **d} for d in fold_metrics])
        
    return metrics, best_state_dict, normalizer


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(device)

    out_path = pathlib.Path(args.output).joinpath(coolname.generate_slug())
    out_path.mkdir(parents=True, exist_ok=True)
    
    # experiment parameters
    exp_params = vars(args)
    exp_params["clinical"] = exp_params["clinical"].split(",")
    exp_params["extractors"] = exp_params["extractors"].split(",")

    # construct data loaders
    internal_loader = DataLoader(Data(base_path="./", cohort=args.internal), split="training")
    external_loader = [DataLoader(Data(base_path="./", cohort=n), split="test") for n in args.external.split(",")]
    
    # data loading
    print("loading cohorts...")
    internal_loader.load()
    for external in external_loader:
        external.load()

    # data preprocessing
    print("preprocess data...")
    internal_loader.preprocess(exp_params, args.task)
    for external in external_loader:
        external.preprocess(exp_params, args.task)

    # fit model
    print("fitting model")
    metrics, best_state_dict, normalizer = bootstrap_training(exp_params, internal_loader, external_loader, device)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='path to save model files')
    parser.add_argument('--internal', type=str, default="radcure", help='name of internal dataset to use for pre-training and training')
    parser.add_argument('--external', type=str, default="artix,headneckpetct,hecktor,radcure", help="comma-separated list of external datasets to use for external validation")
    parser.add_argument('--split_loc', action='store_true', help="wether to split RADCURE test into different localisations for evaluation")
    parser.add_argument('--bootstrap', type=int, default=1, help='number of bootstrap iterations')
    parser.add_argument('--task', type=str, required=True, choices=["rfs_2", "rfs_5"], help="task to train model on")
    parser.add_argument('--pretraining', default=None, choices=["cox+protonet", "protonet"], help="which method to use for pre-training")
    parser.add_argument('--modality', type=str, required=True, choices=["both", "image", "clinical"],  help="which modality to use")
    parser.add_argument('--clinical', type=str, default="", help="comma-separated list of clinical variables to use")
    parser.add_argument('--extractors', type=str, default="ct-fm", help="comma-separated list of extractors to use")
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
    parser.add_argument('--freeze', action='store_true', help="whether to freeze backbone weights during fine-tuning")
    parser.add_argument('--mean_reg_lambda', type=float, default=0., help="lambda parameter for mean divergence regularization loss")
    parser.add_argument('--variance_reg_lambda', type=float, default=0., help="lambda parameter for variance divergence regularization loss")
    parser.add_argument('--class_weights', action='store_true', help="whether to use class weighting in the loss to counter class imbalance")
    parser.add_argument('--undersampling', action='store_true', help="whether to apply data undersampling to the training data")
    parser.add_argument('--gpu', type=str, default="", help='GPUs to use')
    args = parser.parse_args()

    print("Script arguments:")
    for k, v in vars(args).items():
        print(f"    {k}: {v}")

    main(args)

    print("done.")
