import seaborn as sns
import pathlib
import json
import tqdm
import os
from matplotlib import pyplot as plt
import pandas

if __name__ == "__main__":
    base_path = r"C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\experiments"
    out_path = r"C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\tmp\plots\training metrics"
    AVERAGE_BOOTSTRAP = True
    df = []
    for path_ in pathlib.Path(base_path).glob("*"):
        print(path_.name)
        
        try:
            if int(path_.name) < 16 or int(path_.name) == 21:
                continue
        except ValueError:
            continue

        for run in tqdm.tqdm(path_.iterdir()):
            if not(run.joinpath("train_metrics.csv").exists()):
                continue
            
            metrics = pandas.read_csv(run.joinpath("train_metrics.csv")).drop(columns="Unnamed: 0")
            metrics["name"] = run.name
            metrics["exp"] = path_.name

            with open(run.joinpath("params.json"), "r") as f:
                p = json.load(f)
                metrics["classifier"] = p["classifier"]

            # add to df
            df.extend(metrics.to_dict(orient="records"))

    dff = pandas.DataFrame(df)
    for metric in dff["metric"].unique():
        print(metric)
        for exp in tqdm.tqdm(dff["exp"].unique(), ncols=50):
            exp_df = dff[(dff["exp"] == exp) & (dff["metric"] == metric)]
            print(exp)

            cols = len(exp_df["classifier"].unique())
            rows = max([len(exp_df[exp_df["classifier"] == c]["name"].unique()) for c in exp_df["classifier"].unique()])
            fig, axes = plt.subplots(rows, cols, figsize=(30, 40))
            
            for c, classifier in enumerate(exp_df["classifier"].unique()):
                for r, name in enumerate(exp_df[exp_df["classifier"] == classifier]["name"].unique()):
                    data = exp_df[(exp_df["classifier"] == classifier) & (exp_df["name"] == name)]
                    if AVERAGE_BOOTSTRAP:   # average bootstraps                    
                        data = data.drop(columns="bootstrap")
                        if cols > 1:
                            sns.lineplot(ax=axes[r,c], data=data, x="step", y="value", hue="split")
                        else:
                            sns.lineplot(ax=axes[r], data=data, x="step", y="value", hue="split")
                    else:
                        if cols > 1:
                            sns.lineplot(ax=axes[r,c], data=data, x="step", y="value", hue="bootstrap")
                        else:
                            sns.lineplot(ax=axes[r], data=data, x="step", y="value", hue="bootstrap")
                    
                    if cols > 1:
                        axes[r,c].set_title(f"{exp} | {classifier} | {name}")
                    else:
                        axes[r].set_title(f"{exp} | {classifier} | {name}")
            plt.subplots_adjust(hspace=0.3)
            plt.savefig(os.path.join(out_path, f"{metric}-{exp}.png"))

    print("Done")
