import os
import argparse
import json

import numpy as np
import pandas as pd
import itertools
import sklearn

import torch

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm

def truncate(s: str, max_len=40):
    s = str(s)
    if len(str(s)) <= max_len:
        return s
    else:
        delta = (max_len - 3) // 2
        return f"{s[:delta]}...{s[-delta:]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str,
                        help="Path to load data from, should be the output folder of a benchmark experiment")
    parser.add_argument("--save", type=str,
                        help="Path to save results to")
    parser.add_argument("--experiment_only", action="store_true", default=False,
                        help="If true, only run the experiment and not the analysis")
    # parser.add_argument("--df", type=str, default=None)
    args = parser.parse_args()

    SEED = 2024
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    if os.path.exists(args.save):
        print(f"WARNING: {args.save} already exists")
    os.makedirs(args.save, exist_ok=True)

    # Create a master dataframe of test individuals
    train = pd.read_csv(os.path.join(args.load, "data", "train.csv"), index_col=0)
    val = pd.read_csv(os.path.join(args.load, "data", "val.csv"), index_col=0)
    test = pd.read_csv(os.path.join(args.load, "data", "test.csv"), index_col=0)

    # Get dataset
    if "malaria" in args.load:
        DATASET = "malaria"
    elif "pddb" in args.load:
        DATASET = "pddb"
    elif "isic" in args.load:
        DATASET = "isic"
    else:
        raise Exception

    # Get demographics columns and types
    if DATASET == "malaria":
        DEMOGRAPHIC_COLS = {
            "Country": "categorical",
            "Asexual.stage..hpi.": "categorical",
            "Kmeans.Grp": "categorical",
        }
    elif DATASET == "pddb":
        DEMOGRAPHIC_COLS = {
            "createdOn": "continuous",
            "appVersion": "categorical",
            "phoneInfo": "categorical",
            "medTimepoint": "categorical",
            "age": "continuous",
            "are-caretaker": "categorical",
            "deep-brain-stimulation": "categorical",
            "diagnosis-year": "continuous",
            "education": "categorical",
            "employment": "categorical",
            "gender": "categorical",
            "healthcare-provider": "categorical",
            "home-usage": "categorical",
            "last-smoked": "continuous",
            "maritalStatus": "categorical",
            "medical-usage": "categorical",
            "medical-usage-yesterday": "categorical", 
            "medication-start-year": "continuous",
            "onset-year": "continuous",
            "packs-per-day": "continuous",
            "past-participation": "categorical",
            "phone-usage": "categorical",
            "race": "categorical",
            "smartphone": "categorical",
            "smoked": "categorical",
            "surgery": "categorical",
            "video-usage": "categorical",
            "years-smoking": "continuous",
        }
    elif DATASET == "isic":
        DEMOGRAPHIC_COLS = {
            "attribution": "categorical",
            "acquisition_day": "continuous",
            "age_approx": "continuous",
            "anatom_site_general": "categorical",
            "clin_size_long_diam_mm": "continuous",
            "concomitant_biopsy": "categorical",
            "dermoscopic_type": "categorical",
            "diagnosis_confirm_type": "categorical",
            "family_hx_mm": "categorical",
            "fitzpatrick_skin_type": "categorical",
            "image_type": "categorical", 
            "mel_class": "categorical",
            "mel_mitotic_index": "categorical",
            "mel_thick_mm": "continuous",
            "mel_type": "categorical",
            "mel_ulcer": "categorical",
            "melanocytic": "categorical",
            "nevus_type": "categorical",
            "personal_hx_mm": "categorical",
            "sex": "categorical",
        }
    else:
        raise ValueError
    
    MODEL_NAME = "lightgbm" if "malaria" in args.load else "base"

    # Add model data to test
    test_predictions = torch.load(os.path.join(args.load, "models", MODEL_NAME, "test_predictions.pt")).cpu().numpy()
    test_y = torch.load(os.path.join(args.load, "models", MODEL_NAME, "test_y.pt")).cpu().numpy()

    for i in range(np.shape(test_predictions)[1]):
        test[f"model::test_predictions_{i}"] = test_predictions[:, i]
    test["model::test_y"] = test_y

    ood_detectors_performance = {"success": list(), "fail": list()}

    # Add OOD detector data to test
    for ood_detector in sorted(os.listdir(os.path.join(args.load, "ood_detectors"))):
        ood_detector_path = os.path.join(args.load, "ood_detectors", ood_detector)
        if not os.path.isdir(ood_detector_path):
            continue
        
        with open(os.path.join(ood_detector_path, "id_ood_indices.json"), "r") as f:
            id_ood_indices = json.loads(f.read())

        # Sanity check to make sure ID/OOD indices are handled correctly
        id_indices = id_ood_indices["id_indices"]
        ood_indices = id_ood_indices["ood_indices"]

        id_obs = pd.read_csv(os.path.join(ood_detector_path, "id_obs.csv"), index_col=0)
        ood_obs = pd.read_csv(os.path.join(ood_detector_path, "ood_obs.csv"), index_col=0)

        assert set.union(set(id_indices), set(ood_indices)) == set(range(len(test)))
        assert set.intersection(set(id_indices), set(ood_indices)) == set()

        test[f"ood_detector::{ood_detector}::prediction"] = "OOD"
        test.iloc[id_ood_indices["id_indices"], test.columns.get_loc(f"ood_detector::{ood_detector}::prediction")] = "ID"

        assert set(id_obs.index) == set(test.loc[test[f"ood_detector::{ood_detector}::prediction"] == "ID", :].index)
        assert set(ood_obs.index) == set(test.loc[test[f"ood_detector::{ood_detector}::prediction"] == "OOD", :].index)

        with open(os.path.join(ood_detector_path, "performance.json"), "r") as f:
            performance = json.loads(f.read())

        if performance["main"]["ID"]["AUROC"] - performance["main"]["OOD"]["AUROC"] > 0.05: # at least 0.05 improvement in AUROC on avg
            ood_detectors_performance["success"].append(ood_detector)
        else:
            ood_detectors_performance["fail"].append(ood_detector)

    # Save the dataframe
    if not args.experiment_only:
        SAVE_DATA_PATH = os.path.join(args.save, "data")
        os.makedirs(SAVE_DATA_PATH, exist_ok=True)

        train.to_csv(os.path.join(SAVE_DATA_PATH, "train.csv"))
        val.to_csv(os.path.join(SAVE_DATA_PATH, "val.csv"))
        test.to_csv(os.path.join(SAVE_DATA_PATH, "test.csv"))

        with open(os.path.join(SAVE_DATA_PATH, "ood_detectors_performance.json"), "w") as f:
            f.write(json.dumps(ood_detectors_performance, indent=2))

    # Do subsequent analyses separately between successful and unsuccessful OOD detectors

    # Do analysis: For each OOD detector, is there a significant difference in the distribution of each column?
    if not args.experiment_only:
        SAVE_PLOT_PATH = os.path.join(args.save, "plots")
        os.makedirs(SAVE_PLOT_PATH, exist_ok=True)

        # Make train plots
        ncol = len(DEMOGRAPHIC_COLS)
        nrow = 1
        figsize = (ncol * 4, nrow * 5)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

        for j, (demographic_var, demographic_var_type) in tqdm.tqdm(enumerate(DEMOGRAPHIC_COLS.items()), desc="Plot train demographics", total=len(DEMOGRAPHIC_COLS)):

            ax = axes[j]
        
            if demographic_var_type == "categorical":
                sns.countplot(
                    x = demographic_var, stat = "count",
                    data = train, ax = ax,
                )
                ax.set_xticklabels([truncate(s.get_text()) for s in ax.get_xticklabels()], rotation=45, ha="right")
                ax.set(xlabel=None, ylabel=None)

            elif demographic_var_type == "continuous":
                sns.histplot(
                    x = demographic_var,
                    bins = 15, alpha = 0.5, kde = False,
                    data = train, ax = ax,
                )
                ax.set(xlabel=None, ylabel=None)

        for ax, demographic_var in zip(axes, list(DEMOGRAPHIC_COLS.keys())):
            ax.set_title(demographic_var, size=12, pad=10)

        SUPTITLE_FONTSIZE = 24
        REL_OFFSET = (SUPTITLE_FONTSIZE / 35) / figsize[1] # in inches
        
        plt.tight_layout()

        plt.suptitle(f"Demographics distributions in train set", fontsize=SUPTITLE_FONTSIZE, y=1+REL_OFFSET)
        plt.savefig(os.path.join(SAVE_PLOT_PATH, f"train.pdf"), dpi=400, bbox_inches="tight")
        
        plt.close("all")
    
    # Make test plots
    # (N_ood_detector, N_demographic_col) plot
    # one for success, one for fail


    for key, ood_detectors in ood_detectors_performance.items(): # key in ("success", "fail")

        if args.experiment_only:
            continue

        for countplot_dodge in (True, False):
            
            ncol = len(DEMOGRAPHIC_COLS)
            nrow = len(ood_detectors_performance[key])
            figsize = (ncol * 4, nrow * 5)
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

            for i, ood_detector in tqdm.tqdm(enumerate(ood_detectors), desc=f"Plot {key}", total=len(ood_detectors)):
                for j, (demographic_var, demographic_var_type) in enumerate(DEMOGRAPHIC_COLS.items()):
                    ax = axes[i, j]

                    palette = {"ID": "#f4a261", "OOD": "#8e44ad"}

                    if demographic_var_type == "categorical":
                        if countplot_dodge:
                            sns.countplot(
                                x = demographic_var, hue = f"ood_detector::{ood_detector}::prediction", stat = "count" if countplot_dodge else "proportion",
                                data = test, ax = ax, dodge = countplot_dodge, palette = palette, hue_order=["ID", "OOD"],
                            )
                            ax.set_xticklabels([truncate(s.get_text()) for s in ax.get_xticklabels()], rotation=45, ha="right")
                            ax.legend(title="Prediction")
                            ax.set(xlabel=None, ylabel=None)
                        else:
                            sns.countplot(x=demographic_var, hue = f"ood_detector::{ood_detector}::prediction",
                                          data=test, ax=ax, dodge=False, palette=palette, hue_order=["ID", "OOD"])

                            counts = pd.crosstab(test[demographic_var], test[f"ood_detector::{ood_detector}::prediction"])
                            counts = counts.div(counts.sum(axis=1,), axis=0)
                            ax.clear()
                            
                            bottom = [0 for _ in range(len(counts))]
                            for k, hue in enumerate(counts.columns):
                                ax.bar(counts.index, counts[hue], bottom=bottom, label=hue, color=palette[hue])
                                bottom = bottom + counts[hue]
                                
                            ax.set_xticklabels([truncate(s.get_text()) for s in ax.get_xticklabels()], rotation=45, ha="right")
                            ax.legend(title="Prediction")

                    elif demographic_var_type == "continuous":
                        sns.histplot(
                            x = demographic_var, hue = f"ood_detector::{ood_detector}::prediction",
                            bins = 15, alpha = 0.5, kde = False, palette = palette,
                            data = test, ax = ax, legend=True,
                        )
                        ax.legend(
                            handles = [plt.Line2D([0], [0], color=palette[group], lw=4, label=group) for group in palette.keys()],
                            title="Prediction"
                        )
                        ax.set(xlabel=None, ylabel=None)
                    else:
                        raise Exception(f"{demographic_var_type} not recognized")

            # Set the labels
            for ax, ood_detector in zip(axes[:, 0], ood_detectors):
                ax.set_ylabel(ood_detector, rotation=90, size=12, labelpad=20, va="center")
            for ax, demographic_var in zip(axes[0, :], list(DEMOGRAPHIC_COLS.keys())):
                ax.set_title(demographic_var, size=12, pad=10)

            SUPTITLE_FONTSIZE = 24
            REL_OFFSET = (SUPTITLE_FONTSIZE / 35) / figsize[1] # in inches
            
            # plt.tight_layout(rect=[0, 0, 1, 1 - REL_OFFSET])
            plt.tight_layout()

            plt.suptitle(f"Demographics distributions among {key} OOD detectors", fontsize=SUPTITLE_FONTSIZE, y=1+REL_OFFSET)
            save_name = str(key) + ("_dodge" if countplot_dodge else "")
            plt.savefig(os.path.join(SAVE_PLOT_PATH, f"{save_name}.pdf"), dpi=400, bbox_inches="tight")
            
            plt.close("all")
    
    # Does the model perform systematically worse on any of these groups? (Maybe do this after you've checked manually the above)
    SAVE_ANALYSIS_PATH = os.path.join(args.save, "analysis")
    os.makedirs(SAVE_ANALYSIS_PATH, exist_ok=True)

    if DATASET == "isic":
        test_predictions_are_probs = torch.all(torch.abs(torch.sum(torch.tensor(test_predictions), dim=1) - 1.0) < 1e-4)
        if not test_predictions_are_probs:
            test_predictions = torch.nn.Softmax(dim=-1)(torch.tensor(test_predictions)).numpy()
        
        bootstrap_df = {
            "replicate": list(), 
            "group": list(),
            "metric": list(),
            "value": list()
        }
        masks = {
            "under_45": np.array(test["age_approx"] < 45, dtype=bool),
        }
        masks["over_45"] = ~masks["under_45"]
        
        indices = {
            group: np.arange(len(test), dtype=int)[masks[group]] for group in masks.keys()
        }
        for replicate in tqdm.tqdm(list(range(1000)) + ["main",], desc="Bootstrap ISIC age", total=1001):

            if replicate == "main":
                replicate_indices = np.arange(len(test))
            else:
                replicate_indices = np.random.choice(len(test), size=len(test), replace=True)

            replicate_indices = {
                group: [i for i in replicate_indices if i in indices[group]] for group in masks.keys()
            }
            replicate_test_predictions = {
                group: test_predictions[replicate_indices[group], :] for group in masks.keys()
            }
            replicate_test_y = {
                group: test_y[replicate_indices[group]] for group in masks.keys()
            }
            metrics = ("accuracy", "auroc")
            values = {group: {metric: None for metric in metrics} for group in masks.keys()}
            for group in masks.keys():
                for metric in metrics:
                    if metric == "accuracy":
                        value = np.mean(np.argmax(replicate_test_predictions[group], axis=1).flatten() == replicate_test_y[group].flatten())
                    elif metric == "auroc":
                        value = sklearn.metrics.roc_auc_score(replicate_test_y[group], replicate_test_predictions[group][:, 1])
                    else:
                        raise ValueError
                    
                    value = float(value)

                    values[group][metric] = value
            
            for group in masks.keys():
                for metric in metrics:
                    bootstrap_df["replicate"].append(str(replicate))
                    bootstrap_df["group"].append(group)
                    bootstrap_df["metric"].append(metric)
                    bootstrap_df["value"].append(values[group][metric])
    
        bootstrap_df = pd.DataFrame(bootstrap_df)
        bootstrap_df.to_csv(os.path.join(SAVE_ANALYSIS_PATH, "bootstrap_df.csv"))

        # Make some plots

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            x = "metric", y = "value", hue = "group",
            split = False, data = bootstrap_df, ax = ax
        )
        ax.set_title("Accuracy and AUROC for under/over 45s in test set")
        ax.legend(title="metric")

        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_ANALYSIS_PATH, f"bootstrap_df.pdf"), dpi=400, bbox_inches="tight")
        plt.close("all")
    elif DATASET == "malaria":
        test_predictions_are_probs = torch.all(torch.abs(torch.sum(torch.tensor(test_predictions), dim=1) - 1.0) < 1e-4)
        if not test_predictions_are_probs:
            test_predictions = torch.nn.Softmax(dim=-1)(torch.tensor(test_predictions)).numpy()
        
        bootstrap_df = {
            "replicate": list(), 
            "group": list(),
            "metric": list(),
            "value": list()
        }
        masks = {
            "asexual_stage_other": ~np.array(test["Asexual.stage..hpi."].isin([10, 12, 14, 16]), dtype=bool),
        }
        masks["asexual_stage_10_12_14_16"] = ~masks["asexual_stage_other"]
        
        indices = {
            group: np.arange(len(test), dtype=int)[masks[group]] for group in masks.keys()
        }
        for replicate in tqdm.tqdm(list(range(1000)) + ["main",], desc="Bootstrap Malaria asexual-stage", total=1001):

            if replicate == "main":
                replicate_indices = np.arange(len(test))
            else:
                replicate_indices = np.random.choice(len(test), size=len(test), replace=True)

            replicate_indices = {
                group: [i for i in replicate_indices if i in indices[group]] for group in masks.keys()
            }
            replicate_test_predictions = {
                group: test_predictions[replicate_indices[group], :] for group in masks.keys()
            }
            replicate_test_y = {
                group: test_y[replicate_indices[group]] for group in masks.keys()
            }
            metrics = ("accuracy", "auroc")
            values = {group: {metric: None for metric in metrics} for group in masks.keys()}
            for group in masks.keys():
                for metric in metrics:
                    if metric == "accuracy":
                        value = np.mean(np.argmax(replicate_test_predictions[group], axis=1).flatten() == replicate_test_y[group].flatten())
                    elif metric == "auroc":
                        value = sklearn.metrics.roc_auc_score(replicate_test_y[group], replicate_test_predictions[group][:, 1])
                    else:
                        raise ValueError
                    
                    value = float(value)

                    values[group][metric] = value
            
            for group in masks.keys():
                for metric in metrics:
                    bootstrap_df["replicate"].append(str(replicate))
                    bootstrap_df["group"].append(group)
                    bootstrap_df["metric"].append(metric)
                    bootstrap_df["value"].append(values[group][metric])
    
        bootstrap_df = pd.DataFrame(bootstrap_df)
        bootstrap_df.to_csv(os.path.join(SAVE_ANALYSIS_PATH, "bootstrap_df.csv"))

        # Make some plots

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            x = "metric", y = "value", hue = "group",
            split = False, data = bootstrap_df, ax = ax
        )
        ax.set_title("Accuracy and AUROC for asexual stage HPI 12/14/16 in test set")
        ax.legend(title="metric")

        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_ANALYSIS_PATH, f"bootstrap_df.pdf"), dpi=400, bbox_inches="tight")
        plt.close("all")
    
    print("Done!")