import pathlib
import json
import pandas
import tqdm
import seaborn as sns
import os
from matplotlib import pyplot as plt

AVERAGE_KFOLD = True

base_path = r"C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\experiments"
df = []
for path_ in pathlib.Path(base_path).glob("*"):    
    try:
        if not(int(path_.name) in [78]):
            continue
    except ValueError:
        continue
    
    for run in tqdm.tqdm(path_.iterdir()):
        if not(run.joinpath("metrics.csv").exists()) or not(run.joinpath("params.json").exists()):
            continue
        
        metrics = pandas.read_csv(run.joinpath("metrics.csv")).drop(columns="Unnamed: 0")
        metrics = metrics[metrics["split"].isin(["train", "valid"])]
        metrics["name"] = run.name
        metrics["exp"] = path_.name

        with open(run.joinpath("params.json"), "r") as f:
            p = json.load(f)
            metrics["backbone"] = p["backbone"]
            metrics["dataset"] = p["dataset"]
            metrics["task"] = p["task"]

        # add to df
        df.extend(metrics.to_dict(orient="records"))

df = pandas.DataFrame(df)    

for exp in tqdm.tqdm(df["exp"].unique()):
    print(exp)
    exp_df = df[(df["exp"] == exp) & (df["metric"].isin(["log_loss", "proto_loss", "cox_loss", "cox_protonet_loss"]))]

    dataset = "headneckpetct"
    exp_df = exp_df[exp_df["dataset"] == dataset]

    for task in exp_df["task"].unique():
        print(task)
        for metric in exp_df["metric"].unique():
            print(metric)
            metric_df = exp_df[(exp_df["dataset"] == dataset) & (exp_df["metric"] == metric) & (exp_df["task"] == task)]
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
                        axes[r,c].set_title(f"{exp} | {backbone} | {name[:15]}")
                    else:
                        axes[r].set_title(f"{exp} | {backbone} | {name[:15]}")
            plt.subplots_adjust(hspace=0.3)
            fig.set_size_inches(8*cols, 6*rows)
            plt.savefig(os.path.join(r"C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\tmp", f"{metric}-{dataset}-{task}-{exp}.png"))