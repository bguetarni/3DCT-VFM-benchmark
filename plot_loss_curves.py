import pathlib
import json
import pandas
import tqdm
import seaborn as sns
import os
from matplotlib import pyplot as plt
import argparse

AVERAGE_KFOLD = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    args = parser.parse_args()
    
    base_path = pathlib.Path("./experiments").joinpath(args.exp)
    if not(base_path.exists()):
        print(f"Experiment {args.exp} does not exist.")
        exit()
    
    df = []
    for run in tqdm.tqdm(list(base_path.iterdir())[:8], ncols=100):
        if not(run.joinpath("metrics.csv").exists()) or not(run.joinpath("params.json").exists()):
            continue
        
        metrics = pandas.read_csv(run.joinpath("metrics.csv")).drop(columns="Unnamed: 0")
        metrics = metrics[metrics["split"].isin(["train", "valid"])]
        metrics["name"] = run.name
        metrics["exp"] = base_path.name

        with open(run.joinpath("params.json"), "r") as f:
            p = json.load(f)
            metrics["backbone"] = p["backbone"]
            metrics["task"] = p["task"]

        # add to df
        df.extend(metrics.to_dict(orient="records"))

    df = pandas.DataFrame(df)
    df = df[df["metric"].isin(["log_loss", "proto_loss", "cox_loss", "cox_protonet_loss"])]
    for task in df["task"].unique():
        print(task)
        for metric in df["metric"].unique():
            print(metric)
            metric_df = df[(df["metric"] == metric) & (df["task"] == task)]
            cols = len(metric_df["backbone"].unique())
            rows = max([len(metric_df[metric_df["backbone"] == c]["name"].unique()) for c in metric_df["backbone"].unique()])
            fig, axes = plt.subplots(rows, cols)
            
            for c, backbone in enumerate(metric_df["backbone"].unique()):
                for r, name in enumerate(metric_df[metric_df["backbone"] == backbone]["name"].unique()):
                    data = metric_df[(metric_df["backbone"] == backbone) & (metric_df["name"] == name)]
                    if AVERAGE_KFOLD:   # average kfolds                  
                        data = data.drop(columns="run")
                        if cols > 1:
                            sns.lineplot(ax=axes[r,c], data=data, x="step", y="value", hue="split")
                        else:
                            sns.lineplot(ax=axes[r], data=data, x="step", y="value", hue="split")
                    else:
                        if cols > 1:
                            sns.lineplot(ax=axes[r,c], data=data, x="step", y="value", hue="run")
                        else:
                            sns.lineplot(ax=axes[r], data=data, x="step", y="value", hue="run")
                    
                    if cols > 1:
                        axes[r,c].set_title(f"{base_path.name} | {backbone} | {name[:15]}")
                    else:
                        axes[r].set_title(f"{base_path.name} | {backbone} | {name[:15]}")
            plt.subplots_adjust(hspace=0.3)
            fig.set_size_inches(8*cols, 6*rows)
            plt.savefig(os.path.join("./tmp", f"{metric}-{task}-{base_path.name}.png"))
