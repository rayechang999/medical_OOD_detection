import os
import argparse
import json
import sys

import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
import time

def mean(x):
    try:
        return np.mean(x)
    except ValueError: # if an array is empty
        return float("nan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--df", type=str, default=None)
    args = parser.parse_args()
    
    BASE_PATH = os.path.join("/path", "to", "projects")
    args.load = os.path.join(BASE_PATH, args.load)
    args.save = os.path.join(BASE_PATH, args.save)

    if os.path.exists(args.save):
        print(f"WARNING: {args.save} already exists")
    os.makedirs(args.save, exist_ok=True)

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=["ood_detector", "replicate", "dist", "metric", "value"])

    ood_detectors = list()

    for ood_detector in tqdm.tqdm(os.listdir(os.path.join(args.load, "ood_detectors")), desc="Fill dataframes"):
        if args.df is not None:
            continue

        ood_detector_path = os.path.join(args.load, "ood_detectors", ood_detector)
        if not os.path.isdir(ood_detector_path):
            continue

        ood_detectors.append(ood_detector)

        performance_path = os.path.join(ood_detector_path, "performance.json")
        if not os.path.exists(performance_path):
            print(f"WARNING: {performance_path} does not exist")
            continue
        with open(performance_path, "r") as f:
            performance = json.load(f)

        replicates = list(performance.keys())
        dist_types = ["OOD", "ID", "all"]
        metrics = ["accuracy", "AUROC", "num_ood", "num_id", "ood_rate", "id_rate"]
        product = list(itertools.product(replicates, dist_types, metrics))
        temp_df = pd.DataFrame(product, columns=["replicate", "dist", "metric"])
        temp_df["value"] = float("nan")
        temp_df["ood_detector"] = ood_detector

        for replicate in tqdm.tqdm(replicates, desc=f"Fill dataframe {ood_detector}"):
            num_ood = performance[replicate]["metadata"]["num_ood"]
            num_id = performance[replicate]["metadata"]["num_id"]
            ood_rate = num_ood / (num_ood + num_id)
            id_rate = num_id / (num_ood + num_id)
            for dist_type in dist_types:
                for metric in metrics:
                    if metric in ("accuracy", "AUROC"):
                        temp_df.loc[
                            (temp_df["replicate"] == replicate) &
                            (temp_df["dist"] == dist_type) &
                            (temp_df["metric"] == metric),
                            "value"
                        ] = performance[replicate][dist_type][metric]
                    elif metric in ("num_ood", "num_id"):
                        temp_df.loc[
                            (temp_df["replicate"] == replicate) &
                            (temp_df["dist"] == dist_type) &
                            (temp_df["metric"] == metric),
                            "value"
                        ] = performance[replicate]["metadata"][metric]
                    elif metric in ("ood_rate", "id_rate"):
                        rate = ood_rate if metric == "ood_rate" else id_rate
                        temp_df.loc[
                            (temp_df["replicate"] == replicate) &
                            (temp_df["dist"] == dist_type) &
                            (temp_df["metric"] == metric),
                            "value"
                        ] = rate
                    else:
                        raise ValueError(f"Invalid metric: {metric}")


        # Append the temporary DataFrame to the main DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)
        
    # df columns: ood_detector, replicate, dist, metric, value
    if args.df is not None:
        df = pd.read_csv(args.df)
    else:
        df.to_csv(os.path.join(args.save, "df.csv"), index=False)

    
    ENSEMBLE_SIZES = [3, 5]

    MODEL_TYPES = ["base", "instance_classification", "logit_norm", "MixUp"]

    MODELS = list()
    for model in MODEL_TYPES:
        MODELS.append(model) # no ensemble
        for ensemble_size in ENSEMBLE_SIZES:
            MODELS.append(f"{model}_ensemble_{ensemble_size}")
    
    OOD_DETECTORS = ["EBO", "IF", "KNN", "Mahalanobis", "MSP", "MSP_entropy"]

    PALETTE = {"all": "#808080", "ID": "#f4a261", "OOD": "#8e44ad"}

    # ---- Make plots of all ----
    all_save_path = os.path.join(args.save, "all")
    os.makedirs(all_save_path, exist_ok=True)
    
    ood_detectors = [f"{model}_{ood_detector}" for model, ood_detector in itertools.product(MODELS, OOD_DETECTORS)]

    # Violin plot of OOD rate for each condition
    if True:
        save_path = os.path.join(all_save_path, f"ood_rate.png")

        plt.figure(figsize=(0.5 * len(MODELS) * len(OOD_DETECTORS), 6))
        sns.violinplot(
            x="ood_detector",
            y="value",
            data = df.loc[(df["dist"] == "ID") & (df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "ood_rate"), :], # id_rate, ood_rate are duplicated across dist==ID and ==OOD
            order = ood_detectors
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("OOD detector")
        plt.ylabel("OOD rate")
        plt.title(f"OOD rate")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close("all")
    
    # Violin plot of ID AUROC for each condition
    if True:
        save_path = os.path.join(all_save_path, f"auroc.png")

        plt.figure(figsize=(1 * len(MODELS) * len(OOD_DETECTORS), 6))
        sns.violinplot(
            x="ood_detector",
            y="value",
            hue="dist",
            data = df.loc[(df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "AUROC"), :],
            order = ood_detectors,
            hue_order = ["all", "ID", "OOD"],
            palette = PALETTE,
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("OOD detector")
        plt.ylabel("AUROC")
        plt.title(f"AUROC")
        plt.legend(title="Test subset")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close("all")


    # ---- Make plots by OOD detector ----
    for ood_detector in OOD_DETECTORS:

        detector_save_path = os.path.join(args.save, "by_ood_detector")
        os.makedirs(detector_save_path, exist_ok=True)

        ood_detectors = [f"{model}_{ood_detector}" for model in MODELS]

        # Violin plot of OOD rate for each condition
        if True:
            save_path = os.path.join(detector_save_path, f"ood_rate_{ood_detector}.png")

            plt.figure(figsize=(1 * len(ood_detectors), 6))
            sns.violinplot(
                x="ood_detector",
                y="value",
                data = df.loc[(df["dist"] == "ID") & (df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "ood_rate"), :], # id_rate, ood_rate are duplicated across dist==ID and ==OOD
                order = ood_detectors,
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("OOD detector")
            plt.ylabel("OOD rate")
            plt.title(f"OOD rate, {ood_detector}")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        
        # Violin plot of ID AUROC for each condition
        if True:
            save_path = os.path.join(detector_save_path, f"auroc_{ood_detector}.png")

            plt.figure(figsize=(1 * len(ood_detectors), 6))
            sns.violinplot(
                x="ood_detector",
                y="value",
                hue="dist",
                data = df.loc[(df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "AUROC"), :],
                order = ood_detectors,
                hue_order = ["all", "ID", "OOD"],
                palette = PALETTE,
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("OOD detector")
            plt.ylabel("AUROC")
            plt.title(f"AUROC, {ood_detector}")
            plt.legend(title="Test subset")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")

    # ---- Make plots by model ----
    for model in MODEL_TYPES:

        models_save_path = os.path.join(args.save, "by_model")
        os.makedirs(models_save_path, exist_ok=True)

        ood_detectors = [f"{m}_{ood_detector}" for ood_detector, m in itertools.product(OOD_DETECTORS, [m for m in MODELS if model in m])]

        # Violin plot of OOD rate for each condition
        if True:
            save_path = os.path.join(models_save_path, f"ood_rate_{model}.png")

            plt.figure(figsize=(1 * len(ood_detectors), 6))
            sns.violinplot(
                x="ood_detector",
                y="value",
                data = df.loc[(df["dist"] == "ID") & (df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "ood_rate"), :], # id_rate, ood_rate are duplicated across dist==ID and ==OOD
                order = ood_detectors,
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("OOD detector")
            plt.ylabel("OOD rate")
            plt.title(f"OOD rate, {model}")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        
        # Violin plot of AUROC for each condition
        if True:
            save_path = os.path.join(models_save_path, f"auroc_{model}.png")

            plt.figure(figsize=(1 * len(ood_detectors), 6))
            sns.violinplot(
                x="ood_detector",
                y="value",
                hue="dist",
                data = df.loc[(df["ood_detector"].isin(ood_detectors)) & (df["metric"] == "AUROC"), :],
                order = ood_detectors,
                hue_order = ["all", "ID", "OOD"],
                palette = PALETTE,
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("OOD detector")
            plt.ylabel("AUROC")
            plt.title(f"AUROC, {model}")
            plt.legend(title="Test subset")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")