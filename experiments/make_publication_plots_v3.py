# Make the publication plots for the paper

import os
import argparse
import json

import scipy
import numpy as np
import pandas as pd
import sklearn

import statsmodels.stats.multitest # for BH

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_venn
import seaborn as sns

import statannotations
from statannotations.Annotator import Annotator

import tqdm

import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str,
                        help="Path to save results to")
    args = parser.parse_args()

    # Preprocess paths
    BASE_PATH = os.path.join("/path", "to", "projects")
    args.save = os.path.join(BASE_PATH, args.save)

    os.makedirs(args.save, exist_ok=True)

    PATHS = {
        "pddb": {
            "benchmark": {
                "raw": os.path.join(BASE_PATH, "benchmark_pddb_ensemble_final"),
                "pooled": os.path.join(BASE_PATH, "maxpool_benchmark_pddb_ensemble_final"),
            },
            "plot": {
                "raw": os.path.join(BASE_PATH, "plot_benchmark_pddb_ensemble_final"),
                "pooled": os.path.join(BASE_PATH, "plot_maxpool_benchmark_pddb_ensemble_final"),
            },
            "demographics": {
                "raw": os.path.join(BASE_PATH, "demographics_benchmark_pddb_ensemble_final"),
                "pooled": os.path.join(BASE_PATH, "demographics_benchmark_pddb_ensemble_final_maxpool"),
            }
        },
        "malaria": {
            "benchmark": os.path.join(BASE_PATH, "benchmark_malaria_ensemble"),
            "plot": os.path.join(BASE_PATH, "plot_benchmark_malaria_ensemble"),
            "demographics": os.path.join(BASE_PATH, "demographics_benchmark_malaria_ensemble"),
        },
        "isic": {
            "benchmark": os.path.join(BASE_PATH, "benchmark_isic_ensemble"),
            "plot": os.path.join(BASE_PATH, "plot_benchmark_isic_ensemble"),
            "demographics": os.path.join(BASE_PATH, "demographics_benchmark_isic_ensemble"),
        },
    }

    FIGURES_TO_MAKE = [
        # "1-a-default", "1-a-idood",

        # "2-a",

        # "3-a",
        # "3-b",
        "3",

        # "4",
        # "5",

        # "T-1",
        # "T-2", # table 1...
        # "metrics",
    ]

    if "1-a-default" in FIGURES_TO_MAKE:
        print(f"Making figure 1-a-default")

        # Use ISIC logit norm KNN all
        df = pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv"))

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.violinplot(
            x="ood_detector",
            y="value",
            data = df.loc[(df["dist"] == "all") & (df["ood_detector"] == "logit_norm_KNN") & (df["metric"] == "AUROC"), :],
            ax = ax
        )
        ax.set_ylabel("AUROC", fontsize=20)
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.legend().remove()
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save, "1-a-default.svg"), format="svg", transparent=True)

        plt.close("all")

    if "1-a-idood" in FIGURES_TO_MAKE:
        print(f"Making figure 1-a-idood")

        # Use ISIC logit norm KNN ID vs. OOD
        df = pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv"))

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.violinplot(
            x="dist",
            y="value",
            hue = "dist",
            data = df.loc[(df["dist"] != "all") & (df["ood_detector"] == "logit_norm_KNN") & (df["metric"] == "AUROC"), :],
            ax = ax,
            palette = {"all": "#808080", "ID": "#f4a261", "OOD": "#8e44ad"},
            order = ["ID", "OOD"]
        )
        ax.set_ylabel("AUROC", fontsize=20)
        ax.set_xlabel("")
        ax.set_yticklabels([])
        # ax.set_xticklabels([])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        ax.legend().remove()
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save, "1-a-idood.svg"), format="svg", transparent=True)

        plt.close("all")
    
    if "2-a" in FIGURES_TO_MAKE:
        print(f"Making figure 2-a")

        dfs = {
            "isic": pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv")),
            "pddb": pd.read_csv(os.path.join(PATHS["pddb"]["plot"]["pooled"], "df.csv")),
            "malaria": pd.read_csv(os.path.join(PATHS["malaria"]["plot"], "df.csv")),
        }
        
        # ---- First, make the dataframe with p-values ----
        print("Loading df")
        df = pd.read_csv(os.path.join(BASE_PATH, "publication_plots_v2", "T-1.csv"))

        TEST_MODE = "two-sided"

        if "2-a.csv" not in os.listdir(args.save):
            table = pd.DataFrame(
                index = df["ood_detector"].unique(),
                columns = ["dataset", "OOD_ID_variance_logfc", "levene_stat", "p_value", "p_value_adj"],
            )
            results = list()
            for dataset in ("isic", "pddb", "malaria"):
                for ood_detector in tqdm.tqdm(df["ood_detector"].unique(), desc=f"Dataset {dataset}", total=len(df["ood_detector"].unique())):
                    data = df.loc[(df["ood_detector"] == ood_detector) & (df["dataset"] == dataset) & (df["metric"] == "AUROC") & (df["replicate"] != "main"), :]

                    id_auroc = data.loc[data["dist"] == "ID", "value"].to_numpy()
                    ood_auroc = data.loc[data["dist"] == "OOD", "value"].to_numpy()

                    # Create boolean mask for non-NaN values in both arrays
                    valid_mask = ~np.isnan(id_auroc) & ~np.isnan(ood_auroc)

                    # Then you can use this mask to select only values where both are valid
                    id_auroc = id_auroc[valid_mask]
                    ood_auroc = ood_auroc[valid_mask]

                    if len(id_auroc) == 0 or len(ood_auroc) == 0:
                        p_value = float("nan")
                    else:
                        if TEST_MODE == "greater":
                            p_value = np.mean(id_auroc <= ood_auroc)
                        elif TEST_MODE == "less":
                            p_value = np.mean(id_auroc >= ood_auroc)
                        elif TEST_MODE == "two-sided":
                            if np.mean(id_auroc - ood_auroc) > 0:
                                p_value = 2 * np.mean(id_auroc <= ood_auroc)
                            else:
                                p_value = 2 * np.mean(id_auroc >= ood_auroc)
                        else:
                            raise ValueError(f"Unknown test mode: {TEST_MODE}")

                    results.append({
                        "dataset": dataset,
                        "ood_detector": ood_detector,
                        "ID_OOD_diff": np.mean(id_auroc - ood_auroc),
                        "p_value": p_value,
                    })
            
            table = pd.DataFrame(results)

            valid_mask = table["ID_OOD_diff"].notna() & table["p_value"].notna()
            _, p_value_adj, _, _ = statsmodels.stats.multitest.multipletests(
                table.loc[valid_mask, "p_value"].to_numpy(),
                alpha=0.05,
                method="fdr_bh",
            )
            table["p_value_adj"] = float("nan")
            table.loc[valid_mask, "p_value_adj"] = p_value_adj

            table.to_csv(os.path.join(args.save, "2-a.csv"), index=False)
        else:
            print("Loading table from file")
            table = pd.read_csv(os.path.join(args.save, "2-a.csv"))
        
        # Proportion of significant p-values
        print(f"Proportion of significant p-values: {(table['p_value_adj'] < 0.05).mean()}")

        # ---- Now make the plots ----

        def score_ood_detector(dfs, ood_detector) -> float:
            FILTER_OOD_PER_DATASET = True
            
            for dataset in (["isic", "pddb", "malaria", "mean"] if FILTER_OOD_PER_DATASET else ["mean",]):
                df = dfs[dataset]
                df = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC"), :]
                id_values = df.loc[df["dist"] == "ID", "value"].reset_index(drop=True)
                ood_values = df.loc[df["dist"] == "OOD", "value"].reset_index(drop=True)

                res = np.where(id_values.isna() | ood_values.isna(), np.nan, id_values > ood_values)
                res = res[~np.isnan(res)]

                if len(res) < 3:
                    return float("nan")

            # here, res is now the result for mean, and in no dataset is there < 3 non-nan OOD replicates (for result stability)
            return float(res.mean())
            

        # Columns of dfs: ood_detector,replicate,dist,metric,value
        # df_copies = {dataset: dfs[dataset].copy(deep=True) for dataset in dfs.keys()}
        # for dataset in df_copies.keys():
        #     df_copies[dataset]["value"] = df_copies[dataset].groupby("metric")["value"].transform(lambda x: x - x.mean())

        dfs["mean"] = pd.concat([dfs["isic"], dfs["pddb"], dfs["malaria"]], axis=0)
        dfs["mean"].reset_index(drop=True, inplace=True)

        ood_detectors_all = [s for s in list(sorted(dfs["mean"]["ood_detector"].unique())) if "ensemble" not in s]

        scores = {
            # "isic": {ood_detector: score_ood_detector(dfs["isic"], ood_detector) for ood_detector in ood_detectors_all},
            # "pddb": {ood_detector: score_ood_detector(dfs["pddb"], ood_detector) for ood_detector in ood_detectors_all},
            # "malaria": {ood_detector: score_ood_detector(dfs["malaria"], ood_detector) for ood_detector in ood_detectors_all},
            "mean": {ood_detector: score_ood_detector(dfs, ood_detector) for ood_detector in ood_detectors_all},
        }

        top_3 = sorted((item for item in scores["mean"].items() if not np.isnan(item[1])), key=lambda x: x[1], reverse=True)[:3]
        bottom_3 = sorted((item for item in scores["mean"].items() if not np.isnan(item[1])), key=lambda x: x[1])[:3]

        for dataset_idx, dataset in enumerate(["isic", "pddb", "malaria"]):

            fig, axes = plt.subplots(6, 2, figsize=(2, 7), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.4})
            fig.subplots_adjust(left=0.20)
            fig_name = f"2-a_{dataset}"

            for ranking in ("top", "bottom"):
                if ranking == "top":
                    ood_detectors = [item[0] for item in top_3]
                else:
                    ood_detectors = [item[0] for item in bottom_3]
                
                print(f"{ranking} OOD detectors: {ood_detectors}")
                
                for i, ood_detector in enumerate(ood_detectors):

                    ood_detector_name = ood_detector.replace("_", " ")
                    for s in ["base", "logit norm", "instance classification", "MixUp"]:
                        ood_detector_name = ood_detector_name.replace(s, f"{s} +")
                    ood_detector_name = ood_detector_name.replace("base", "CE")
                    ood_detector_name = ood_detector_name.replace("instance classification", "IC")
                    ood_detector_name = ood_detector_name.replace("MSP entropy", "MSP ent.")

                    print(i if ranking == "top" else 6 - (i + 1), ood_detector)

                    # Make plot of AUROC
                    data = dfs[dataset].loc[(dfs[dataset]["dist"] != "all") & (dfs[dataset]["ood_detector"] == ood_detector) & (dfs[dataset]["metric"] == "AUROC"), :]
                    ax = axes[i if ranking == "top" else 6 - (i + 1), 0]
                    sns.violinplot(
                        x="dist",
                        y="value",
                        hue = "dist",
                        data = data,
                        ax = ax,
                        order = ["ID", "OOD"],
                        hue_order = ["ID", "OOD"],
                        palette = {"all": "#808080", "ID": "#f4a261", "OOD": "#8e44ad"},
                    )
                    ax.set_ylim(-0.1, 1.1)
                    # ax.set_ylabel("AUROC" if ranking == "top" and i == 0 else "", fontsize=20)
                    # ax.set_xlabel(ood_detector_name, fontsize=20)
                    ax.set_ylabel(ood_detector_name if dataset == "isic" else "", fontsize=8, rotation=90)
                    ax.set_xlabel("")
                    ax.set_title("AUROC" if ranking == "top" and i == 0 else "", fontsize=8)
                    ax.legend().remove()
                    ax.set_xticklabels([])
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                    ax.tick_params(axis="x", length=0)
                    p_value = float(table.loc[(table["ood_detector"] == ood_detector) & (table["dataset"] == dataset), "p_value_adj"].values[0])
                    if p_value < 0.05: # only annotate the ones that are significant; don't draw lines for ns
                        annotator = Annotator(
                            ax,
                            pairs = [("ID", "OOD")],
                            data = data,
                            x = "dist",
                            y = "value",
                        )
                        annotator.set_pvalues([p_value])
                        annotator.annotate()
                    ax.autoscale(False)
                    sns.despine(ax=ax, top=True, right=True, bottom=True)
                    ax.set_facecolor("none")

                    # Make plot of OOD rate
                    ax = axes[i if ranking == "top" else 6 - (i + 1), 1]
                    sns.violinplot(
                        x="ood_detector",
                        y="value",
                        data = dfs[dataset].loc[(dfs[dataset]["ood_detector"] == ood_detector) & (dfs[dataset]["metric"] == "ood_rate") & (dfs[dataset]["dist"] == "ID"), :], # OOD rates are duplicated between dist==ID and OOD
                        ax = ax,
                    )
                    ax.set_ylim(0.0, 0.6)
                    ax.legend().remove()
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                    ax.set_ylabel("")
                    ax.tick_params(axis="x", length=0)
                    # ax.tick_params(axis="y", length=2)
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                    ax.autoscale(False)
                    ax.set_title("OOD rate" if ranking == "top" and i == 0 else "", fontsize=8)
                    sns.despine(ax=ax, top=True, bottom=True)
                    
            fig.patch.set_alpha(0)
        
            plt.tight_layout()
            plt.savefig(os.path.join(args.save, f"{fig_name}.svg"), format="svg", transparent=True)
            plt.close("all")
        
        # Make a separate legend to be saved as its own svg
        fig, ax = plt.subplots(figsize=(1, 1))  # Adjust size as needed

        ood_patch = matplotlib.patches.Patch(color="#8e44ad", label="OOD")
        id_patch = matplotlib.patches.Patch(color="#f4a261", label="ID")

        legend = ax.legend(handles=[id_patch, ood_patch], loc="center", frameon=False)
        
        ax.axis("off")
        plt.savefig(os.path.join(args.save, "2-a-legend.svg"), format="svg", transparent=True)

        plt.close("all")

    if "3" in FIGURES_TO_MAKE:
        print("Making figure 3")

        print("Loading df")
        df = pd.read_csv(os.path.join(BASE_PATH, "publication_plots_v2", "T-1.csv"))

        if "3-a.csv" not in os.listdir(args.save):
            table = pd.DataFrame(
                index = df["ood_detector"].unique(),
                columns = ["dataset", "OOD_ID_variance_logfc", "levene_stat", "p_value", "p_value_adj"]
            )
            results = list()
            for dataset in ("isic", "pddb", "malaria"):
                for ood_detector in tqdm.tqdm(df["ood_detector"].unique(), desc=f"Dataset {dataset}", total=len(df["ood_detector"].unique())):
                    data = df.loc[(df["ood_detector"] == ood_detector) & (df["dataset"] == dataset) & (df["metric"] == "AUROC") & (df["replicate"] != "main"), :]

                    id_auroc = data.loc[data["dist"] == "ID", "value"].to_numpy()
                    ood_auroc = data.loc[data["dist"] == "OOD", "value"].to_numpy()

                    # Remove nans (interpretation: only where OOD rate is nonzero)
                    id_auroc = id_auroc[~np.isnan(id_auroc)]
                    ood_auroc = ood_auroc[~np.isnan(ood_auroc)]

                    logfc = np.log(np.var(ood_auroc, ddof=1) / np.var(id_auroc, ddof=1))

                    if len(id_auroc) < 2 or len(ood_auroc) < 2 or np.isnan(logfc) or np.isinf(logfc): # inf handles the case where there's zero variance due to no meaningful OOD rates
                        levene_stat, p_value = float("nan"), float("nan")
                    else:
                        levene_stat, p_value = scipy.stats.levene(id_auroc, ood_auroc)

                    results.append({
                        "dataset": dataset,
                        "ood_detector": ood_detector,
                        "OOD_ID_variance_logfc": logfc,
                        "levene_stat": levene_stat,
                        "p_value": p_value,
                    })
            
            table = pd.DataFrame(results)

            valid_mask = table["OOD_ID_variance_logfc"].notna() & table["p_value"].notna()
            _, p_value_adj, _, _ = statsmodels.stats.multitest.multipletests(
                table.loc[valid_mask, "p_value"].to_numpy(),
                alpha=0.05,
                method="fdr_bh",
            )
            table["p_value_adj"] = float("nan")
            table.loc[valid_mask, "p_value_adj"] = p_value_adj

            table.to_csv(os.path.join(args.save, "3-a.csv"), index=False)
        else:
            print("Loading table from file")
            table = pd.read_csv(os.path.join(args.save, "3-a.csv"))
            # table.set_index("ood_detector", inplace=True)

        # ---- Volcano plot ----
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.scatterplot(
            x = "OOD_ID_variance_logfc",
            y = -np.log10(table["p_value_adj"]),
            data = table,
            ax = ax,
        )
        ax.set_xlabel("OOD vs. ID variance log FC", fontsize=8)
        ax.set_ylabel("-log10(p-value adj.)", fontsize=8)
        # ax.set_ylim(0, 5)
        # ax.set_xlim(-2, 2)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        # Draw a horizontal line at the significance thresohld
        ax.axhline(y=-np.log10(0.05), color='r', linestyle='--')
        sns.despine(ax=ax, top=True, right=True, bottom=True)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save, "3-volcano.svg"), format="svg", transparent=True)

        # For Venn: Use logit norm Mahalanobis on ISIC
        # ---- Venn Diagram ----
        import ood_detectors
        class MatrixDataloader(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return (self.data[idx], self.labels[idx])
        
        model_outputs = dict()
        for i in range(3):
            model_path = os.path.join(PATHS["isic"]["benchmark"], "models", f"logit_norm_ensemble_3_{i}")
            model_outputs[i] = {
                "train_embedding": torch.load(os.path.join(model_path, "train_embedding.pt")),
                "test_embedding": torch.load(os.path.join(model_path, "test_embedding.pt")),
                "train_y": torch.load(os.path.join(model_path, "train_y.pt")),
                "test_y": pd.read_csv(os.path.join(PATHS["isic"]["benchmark"], "data", "test.csv"))
            }
            model_outputs[i]["test_y"] = torch.tensor([(0 if c == "benign" else 1) for c in model_outputs[i]["test_y"]["benign_malignant"]], dtype=torch.long)
            model_outputs[i]["train_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataloader(model_outputs[i]["train_embedding"], model_outputs[i]["train_y"]),
                batch_size=32,
                shuffle=False,
            )
            model_outputs[i]["test_dataloader"] = torch.utils.data.DataLoader(
                MatrixDataloader(model_outputs[i]["test_embedding"], model_outputs[i]["test_y"]),
                batch_size=32,
                shuffle=False,
            )
        detectors = dict()
        for i in range(3):
            ood_detector = ood_detectors.MahalanobisDistance(threshold=None, threshold_quantile=0.95)
            ood_detector.train(model_outputs[i]["train_dataloader"])
            pred = ood_detector.predict(model_outputs[i]["test_dataloader"])
            # Get the indices of pred that are 1
            indices = np.where(pred == 1)[0]

            detectors[i] = {
                "ood_detector": ood_detector,
                "pred": pred,
                "indices": set([int(i) for i in indices]),
            }
        
        # Make venn
        plt.close("all")

        fig = plt.figure(figsize=(3, 3))
        v = matplotlib_venn.venn3(
            [detectors[i]["indices"] for i in range(3)],
            set_labels=["Detector 1", "Detector 2", "Detector 3"]
        )
        # if v.center_label is not None:
        #     v.center_label.set_visible(False)
        
        for text in v.set_labels:
            if text is not None:
                text.set_fontsize(8)

        for text in v.subset_labels:
            if text is not None:
                text.set_fontsize(8)
        plt.title("Overlap in OOD predictions", fontsize=8)
        plt.tight_layout()
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        ax = plt.gca()
        ax.set_facecolor("none")
        plt.savefig(os.path.join(args.save, "3-venn.svg"), format="svg", transparent=True)

        # ---- Make AUROC plot (logit norm Mahalanobis) ----

        isic_df = pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv"))

        detectors = {
            0: {
                "ood_detector": "logit_norm_Mahalanobis",
                "ood_detector_name": "Single",
            },
            1: {
                "ood_detector": "logit_norm_ensemble_3_Mahalanobis",
                "ood_detector_name": "Ensemble 3",
            },
            2: {
                "ood_detector": "logit_norm_ensemble_5_Mahalanobis",
                "ood_detector_name": "Ensemble 5",
            }
        }
        
        for i, detector_dict in detectors.items():
            ood_detector = detector_dict["ood_detector"]
            ood_detector_name = detector_dict["ood_detector_name"]

            fig, axes = plt.subplots(1, 2, figsize=(2.3, 2.5), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.4})
            # Move left a little
            fig.subplots_adjust(left=0.15)
            ax = axes[0]
            
            data = isic_df.loc[(isic_df["ood_detector"] == ood_detector) & (isic_df["metric"] == "AUROC") & (isic_df["dist"] != "all"), :]

            sns.violinplot(
                x = "dist",
                y = "value",
                hue = "dist",
                data = data,
                ax = ax,
                order = ["ID", "OOD"],
                hue_order = ["ID", "OOD"],
                palette = {"all": "#808080", "ID": "#f4a261", "OOD": "#8e44ad"}
            )
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("AUROC", fontsize=8)#, rotation=90)
            ax.set_ylabel("")
            ax.legend().remove()
            ax.set_xticklabels([])
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.tick_params(axis="x", length=0)
            ax.autoscale(False)
            sns.despine(ax=ax, top=True, right=True, bottom=True)
            ax.set_facecolor("none")
        

            ax = axes[1]
            
            data = isic_df.loc[(isic_df["ood_detector"] == ood_detector) & (isic_df["metric"] == "ood_rate") & (isic_df["dist"] == "ID"), :]
            sns.violinplot(
                x = "ood_detector",
                y = "value",
                data = data,
                ax = ax,
            )
            ax.set_ylim(0.0, 0.6)
            ax.legend().remove()
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_xlabel("OOD rate", fontsize=8)#, rotation=90)
            ax.set_ylabel(None)
            ax.tick_params(axis="x", length=0)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.autoscale(False)
            sns.despine(ax=ax, top=True, bottom=True)
                    
            fig.patch.set_alpha(0)

            # Set an overall title
            fig.suptitle(ood_detector_name, fontsize=8, y=0.95)
        
            plt.tight_layout()
            plt.savefig(os.path.join(args.save, f"3-c_{ood_detector}.svg"), format="svg", transparent=True)
            plt.close("all")



    if "4" in FIGURES_TO_MAKE:
        print("Making figure 4")
        dfs = {
            "isic": {
                "train": pd.read_csv(os.path.join(PATHS["isic"]["demographics"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["isic"]["demographics"], "data", "test.csv")),
            },
            "pddb": {
                "train": pd.read_csv(os.path.join(PATHS["pddb"]["demographics"]["pooled"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["pddb"]["demographics"]["pooled"], "data", "test.csv")),
            },
            "malaria": {
                "train": pd.read_csv(os.path.join(PATHS["malaria"]["demographics"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["malaria"]["demographics"], "data", "test.csv")),
            },
        }

        dfs2 = {# from fig 2
            "isic": pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv")),
            "pddb": pd.read_csv(os.path.join(PATHS["pddb"]["plot"]["pooled"], "df.csv")),
            "malaria": pd.read_csv(os.path.join(PATHS["malaria"]["plot"], "df.csv")),
        }

        DEMOGRAPHIC_VARS = {
            "isic": {
                "anatom_site_general": {
                    "type": "categorical",
                    "name": "anatomical site",
                    # "examples_max": 225,
                    "examples": {
                        "example_1": ("instance_classification_KNN", "inst. class. + KNN"),
                        "example_2": ("base_KNN", "CE + KNN"),
                        "example_3": ("base_Mahalanobis", "CE + Mahalanobis"),
                    },
                },
            },
            "malaria": {
                "Country": {
                    "type": "categorical",
                    "name": "country",
                    "examples": {
                        "example_1": ("MixUp_KNN", "MixUp + KNN"),
                        "example_2": ("base_KNN", "CE + KNN"),
                        "example_3": ("MixUp_IF", "MixUp + IF"),
                    }
                },
            },
            "pddb": {
                "age": {
                    "type": "continuous",
                    "name": "age",
                    "examples": {
                        "example_1": ("logit_norm_EBO", "logit norm + EBO"),
                        "example_2": ("base_KNN", "CE + KNN"),
                        "example_3": ("logit_norm_MSP_entropy", "logit norm + MSP ent."),
                    }
                },
            },
        }

        DATASET_TO_LETTER_MAP = {
            "isic": "a",
            "malaria": "b",
            "pddb": "c",
        }
        PALETTE = {"ID": "#f4a261", "OOD": "#8e44ad"}
        
        for dataset in DEMOGRAPHIC_VARS.keys():
            for demographic_var_idx, demographic_var in enumerate(DEMOGRAPHIC_VARS[dataset].keys()):

                fig, axes = plt.subplots(2, 4, figsize=(7, 3))
                if dataset in ("isic", "malaria"):
                    fig.subplots_adjust(top=0.9, bottom=0.3, wspace=0.4)#, left=0.1, right=0.9, wspace=0.4)
                else:
                    fig.subplots_adjust(wspace=0.4)
                
                demographic_var_type = DEMOGRAPHIC_VARS[dataset][demographic_var]["type"]

                if dataset == "isic":
                    nice_dataset_name = "ISIC"
                elif dataset == "malaria":
                    nice_dataset_name = "Malaria"
                elif dataset == "pddb":
                    nice_dataset_name = "Parkinson's"
                
                # Make title plot
                ax = axes[0, 0]
                ax.axis("off")
                ax.text(0.5, 0.5, nice_dataset_name, 
                        horizontalalignment="center", verticalalignment="center",
                        transform=ax.transAxes, fontsize=10, fontweight='bold')

                # Make train plots
                if True:
                    fig_title = f"4-{DATASET_TO_LETTER_MAP[dataset]}_{demographic_var}_train"
                    
                    ax = axes[1, 0]

                    if demographic_var_type == "categorical":
                        data = dfs[dataset]["train"]
                        if demographic_var == "employment":
                            data[demographic_var] = data[demographic_var].fillna("unknown")

                        sns.countplot(
                            x = demographic_var, stat = "count",
                            data = data, ax = ax,
                            order = ["A", "B", "C"] if demographic_var == "Kmeans.Grp" else sorted(data[demographic_var].unique()),
                        )
                        # ax.set_xticklabels([truncate(s.get_text()) for s in ax.get_xticklabels()], rotation=45, ha="right")
                        ax.tick_params(axis='x', rotation=45)
                        for label in ax.get_xticklabels():
                            label.set_ha('right')
                        ax.legend().remove()
                        sns.despine(ax=ax, top=True, right=True)
                        ax.tick_params(axis="x", length=0)
                        ax.set(xlabel=None, ylabel=None)

                    elif demographic_var_type == "continuous":
                        sns.histplot(
                            x = demographic_var,
                            bins = 12 if demographic_var == "age_approx" else 15, alpha = 1, kde = False,
                            data = dfs[dataset]["train"], ax = ax,
                        )
                        ax.set(xlabel=None, ylabel=None)
                        sns.despine(ax=ax, top=True, right=True, bottom=True)
                        if demographic_var == "age":
                            ax.set_xlim(15, 85)
                
                    if demographic_var_idx == 0:
                        ax.set_title("training distribution", fontsize=8)
                    ax.set_ylabel(DEMOGRAPHIC_VARS[dataset][demographic_var]["name"], fontsize=8, rotation=90)
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                    ax.legend().remove()

                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")

                # Make example plots
                for example_num, (ood_detector, ood_detector_name) in DEMOGRAPHIC_VARS[dataset][demographic_var]["examples"].items():
                    
                    # ---- Make model performance ID/OOD AUROC violin plots on top row ----
                    ax = axes[0, int(example_num.split("_")[-1])]
                    sns.violinplot(
                        x="dist",
                        y="value",
                        hue = "dist",
                        data = dfs2[dataset].loc[(dfs2[dataset]["dist"] != "all") & (dfs2[dataset]["ood_detector"] == ood_detector) & (dfs2[dataset]["metric"] == "AUROC"), :],
                        ax = ax,
                        order = ["ID", "OOD"],
                        hue_order = ["ID", "OOD"],
                        palette = {"all": "#808080", "ID": "#f4a261", "OOD": "#8e44ad"},
                    )
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel("AUROC", fontsize=8, rotation=90)
                    ax.set_xlabel("")
                    ax.set_title(ood_detector_name, fontsize=8)
                    ax.legend().remove()
                    ax.set_xticklabels([])
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                    ax.tick_params(axis="x", length=0)
                    ax.autoscale(False)
                    sns.despine(ax=ax, top=True, right=True, bottom=True)
                    ax.set_facecolor("none")

                    # ---- Make demographics plot on bottom row ----
                    ax = axes[1, int(example_num.split("_")[-1])]
                    if demographic_var_type == "continuous":
                        sns.histplot(
                            x = demographic_var, hue = f"ood_detector::{ood_detector}::prediction",
                            bins = 12 if demographic_var == "age_approx" else 15,
                            alpha = 0.5, kde = False, palette = PALETTE,
                            data = dfs[dataset]["test"], ax = ax, legend=True,
                        )
                        ax.legend().remove()
                        sns.despine(ax=ax, top=True, right=True, bottom=True)
                        ax.set(xlabel=None, ylabel=None)
                        if demographic_var == "age":
                            ax.set_xlim(15, 85)
                    elif demographic_var_type == "categorical":
                        data = dfs[dataset]["test"]
                        if demographic_var == "employment":
                            data[demographic_var] = data[demographic_var].fillna("unknown")
                        sns.countplot(
                            x = demographic_var, hue = f"ood_detector::{ood_detector}::prediction", stat = "count",
                            data = data, ax = ax, dodge = True, palette = PALETTE, hue_order=["ID", "OOD"],
                            order = ["A", "B", "C"] if demographic_var == "Kmeans.Grp" else sorted(data[demographic_var].unique()),
                        )
                        ax.tick_params(axis='x', rotation=45)
                        for label in ax.get_xticklabels():
                            label.set_ha('right')
                        sns.despine(ax=ax, top=True, right=True)
                        ax.tick_params(axis="x", length=0)
                        ax.legend().remove()
                        ax.set(xlabel=None, ylabel=None)
                    else:
                        raise ValueError("demographic_var_type must be continuous or categorical")

                    ax.legend().remove()
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")

                
                # plt.tight_layout()
                plt.savefig(os.path.join(args.save, f"4-{DATASET_TO_LETTER_MAP[dataset]}_{demographic_var}.svg"), format="svg", transparent=True)
    
    if "5" in FIGURES_TO_MAKE:
        
        print("Making figure 5")

        PALETTE = {"ID": "#f4a261", "OOD": "#8e44ad"}

        dfs = {
            "isic": {
                "train": pd.read_csv(os.path.join(PATHS["isic"]["demographics"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["isic"]["demographics"], "data", "test.csv")),
                "bootstrap": pd.read_csv(os.path.join(PATHS["isic"]["demographics"], "analysis", "bootstrap_df.csv")),
            },
            "pddb": {
                "train": pd.read_csv(os.path.join(PATHS["pddb"]["demographics"]["pooled"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["pddb"]["demographics"]["pooled"], "data", "test_instance_classification_IF.csv")),
                "bootstrap": pd.read_csv(os.path.join(PATHS["pddb"]["demographics"]["pooled"], "analysis", "bootstrap_df.csv")),
            },
            "malaria": {
                "train": pd.read_csv(os.path.join(PATHS["malaria"]["demographics"], "data", "train.csv")),
                "test": pd.read_csv(os.path.join(PATHS["malaria"]["demographics"], "data", "test.csv")),
                "bootstrap": pd.read_csv(os.path.join(PATHS["malaria"]["demographics"], "analysis", "bootstrap_df.csv")),
            },
        }

        # Make isic plots
        if True:

            fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

            dataset = "isic"
            # ---- Make train plot ----
            fig_title = f"5-a_isic_train"
            ax = axes[0]

            sns.histplot(
                x = "age_approx",
                bins = 12, alpha = 0.5, kde = False,
                data = dfs[dataset]["train"], ax = ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("training distribution", fontsize=8)
            ax.set_ylabel("num. observations", fontsize=8)
            sns.despine(ax=ax, top=True, right=True)
            ax.legend().remove()

            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            # ---- Make OOD plot ----
            fig_title = f"5-a_isic_ood"

            ax = axes[1]
            sns.histplot(
                x = "age_approx", hue = f"ood_detector::MixUp_ensemble_3_Mahalanobis::prediction",
                bins = 12, alpha = 0.5, kde = False, palette = PALETTE,
                data = dfs[dataset]["test"], ax = ax, legend=True,
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.legend().remove()
            ax.set_xlabel("MixUp + Mahalanobis (ensemble 3)", fontsize=8)
            ax.set_ylabel("num. observations", fontsize=8) 
            sns.despine(ax=ax, top=True, right=True)

            fig.patch.set_alpha(0)
            
            ax.set_facecolor("none")

            # ---- Make model bootstrap classification plot ----
            fig_title = f"5-a_isic_model"

            data = dfs[dataset]["bootstrap"].loc[dfs[dataset]["bootstrap"]["metric"] == "auroc", :]
            data["group"] = data["group"].replace({"under_75": "under 75", "over_75": "over 75"})

            ax = axes[2]
            sns.violinplot(
                x = "group", y = "value", hue = "group",
                split = False, ax = ax,
                data = data,
                order = ["under 75", "over 75"],
                hue_order = ["under 75", "over 75"],
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.legend().remove()
            ax.set_xlabel("model performance", fontsize=8)
            ax.set_ylabel("AUROC", fontsize=8)
            sns.despine(ax=ax, top=True, right=True)
            ax.tick_params(axis="x", length=0)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            plt.tight_layout()
            plt.savefig(os.path.join(args.save, "5-a_isic_age_approx.svg"), format="svg", transparent=True)
            plt.close("all")


        # Make malaria plots
        if True:

            fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

            dataset = "malaria"
            # ---- Make train plot ----
            fig_title = f"5-b_malaria_train"
            ax = axes[0]

            dfs[dataset]["test"]["Asexual.stage..hpi."] = dfs[dataset]["test"]["Asexual.stage..hpi."].astype(str)
            hpi_order = [str(i) for i in sorted(dfs[dataset]["test"]["Asexual.stage..hpi."].astype(int).unique())]

            sns.countplot(
                x = "Asexual.stage..hpi.", stat = "count",
                data = dfs[dataset]["train"], ax = ax, order=hpi_order,
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("training distribution", fontsize=8)
            ax.set_ylabel("num. observations", fontsize=8)
            sns.despine(ax=ax, top=True, right=True)
            ax.legend().remove()

            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            # plt.tight_layout()
            # plt.savefig(os.path.join(args.save, f"{fig_title}.svg"), format="svg", transparent=True)

            # plt.close("all")

            # ---- Make OOD plot ----
            fig_title = f"5-b_malaria_ood"

            ax = axes[1]
            sns.countplot(
                x="Asexual.stage..hpi.", hue = f"ood_detector::MixUp_MSP::prediction",
                data=dfs[dataset]["test"], ax=ax, dodge=False, palette=PALETTE,
                hue_order=["ID", "OOD"], order=hpi_order)

            counts = pd.crosstab(dfs[dataset]["test"]["Asexual.stage..hpi."], dfs[dataset]["test"][f"ood_detector::MixUp_MSP::prediction"])
            counts = counts.div(counts.sum(axis=1,), axis=0)
            counts = counts.reindex(hpi_order)
            ax.clear()
            
            bottom = [0 for _ in range(len(counts))]
            for k, hue in enumerate(counts.columns):
                ax.bar(counts.index, counts[hue], bottom=bottom, label=hue, color=PALETTE[hue])
                bottom = bottom + counts[hue]
            
            ax.legend().remove()
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("MixUp + MSP", fontsize=8)
            ax.set_ylabel("proportion", fontsize=8)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            # plt.tight_layout()
            # plt.savefig(os.path.join(args.save, f"{fig_title}.svg"), format="svg", transparent=True)

            # plt.close("all")

            # ---- Make model bootstrap classification plot ----
            fig_title = f"5-b_malaria_model"

            data = dfs[dataset]["bootstrap"].loc[dfs[dataset]["bootstrap"]["metric"] == "auroc", :]
            data["group"] = data["group"].replace(
                {"asexual_stage_10_12_14_16": "10-16 hpi", "asexual_stage_other": "others"}
            )

            ax = axes[2]
            sns.violinplot(
                x = "group", y = "value", hue="group",
                split = False, ax = ax,
                data = data,
                order = ["others", "10-16 hpi"],
                hue_order = ["others", "10-16 hpi"],
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("model performance", fontsize=8)
            ax.set_ylabel("AUROC", fontsize=8)
            ax.legend().remove()
            sns.despine(ax=ax, top=True, right=True)
            ax.tick_params(axis="x", length=0)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            # plt.tight_layout()
            # plt.savefig(os.path.join(args.save, f"{fig_title}.svg"), format="svg", transparent=True)

            # plt.close("all")

            plt.tight_layout()
            plt.savefig(os.path.join(args.save, "5-b_malaria_hpi.svg"), format="svg", transparent=True)
            plt.close("all")

        # Make pddb plots
        if True:

            fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

            dataset = "pddb"
            # ---- Make train plot ----
            fig_title = f"5-c_pddb_train"
            ax = axes[0]

            data = dfs[dataset]["train"]
            data["deep-brain-stimulation"] = data["deep-brain-stimulation"].replace({0: "no DBS", 1: "DBS", float("nan"): "unknown"}).astype(str)

            sns.countplot(
                x = "deep-brain-stimulation", stat = "count",
                data = data, ax = ax, order = ["no DBS", "DBS", "unknown"],
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("training distribution", fontsize=8)
            ax.set_ylabel("num. observations", fontsize=8)
            sns.despine(ax=ax, top=True, right=True)
            ax.legend().remove()

            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            # ---- Make OOD plot ----
            fig_title = f"5-c_pddb_ood"

            data = dfs[dataset]["test"]
            data["deep-brain-stimulation"] = data["deep-brain-stimulation"].replace({0: "no DBS", 1: "DBS", float("nan"): "unknown"}).astype(str)

            ax = axes[1]
            sns.countplot(
                x="deep-brain-stimulation", hue = f"ood_detector::instance_classification_IF::prediction",
                data=data, ax=ax, dodge=False, palette=PALETTE,
                hue_order=["ID", "OOD"], order=["no DBS", "DBS", "unknown"],
            )

            counts = pd.crosstab(dfs[dataset]["test"]["deep-brain-stimulation"], dfs[dataset]["test"][f"ood_detector::instance_classification_IF::prediction"])
            counts = counts.div(counts.sum(axis=1,), axis=0)
            counts = counts.reindex(["no DBS", "DBS", "unknown"])
            ax.clear()
            
            bottom = [0 for _ in range(len(counts))]
            for k, hue in enumerate(counts.columns):
                ax.bar(counts.index, counts[hue], bottom=bottom, label=hue, color=PALETTE[hue])
                bottom = bottom + counts[hue]
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("inst. class. + IF", fontsize=8)
            ax.set_ylabel("proportion", fontsize=8)
            ax.legend().remove()
            sns.despine(ax=ax, top=True, right=True)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            # ---- Make model bootstrap classification plot ----
            fig_title = f"5-c_pddb_model"

            data = dfs[dataset]["bootstrap"].loc[dfs[dataset]["bootstrap"]["metric"] == "auroc", :]
            data["group"] = data["group"].replace(
                {"none": "no DBS", "deep_brain_stimulation": "DBS"}
            )

            ax = axes[2]
            sns.violinplot(
                x = "group", y = "value", hue="group",
                split = False, ax = ax,
                data = data,
                order = ["no DBS", "DBS"],
                hue_order = ["no DBS", "DBS"],
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("model performance", fontsize=8)
            ax.set_ylabel("AUROC", fontsize=8)
            ax.legend().remove()
            sns.despine(ax=ax, top=True, right=True)
            ax.tick_params(axis="x", length=0)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
        
            plt.tight_layout()
            plt.savefig(os.path.join(args.save, "5-c_pddb_dbs.svg"), format="svg", transparent=True)
            plt.close("all")
    
    if "T-1" in FIGURES_TO_MAKE:
        print(f"Making Table 1")

        dfs = {
            "isic": pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv")),
            "pddb": pd.read_csv(os.path.join(PATHS["pddb"]["plot"]["pooled"], "df.csv")),
            "malaria": pd.read_csv(os.path.join(PATHS["malaria"]["plot"], "df.csv")),
        }

        # Columns of dfs: ood_detector,replicate,dist,metric,value
        for dataset in dfs.keys():
            dfs[dataset]["dataset"] = dataset

        dfs["mean"] = pd.concat([dfs["isic"], dfs["pddb"], dfs["malaria"]], axis=0)
        dfs["mean"].reset_index(drop=True, inplace=True)

        dfs["mean"].to_csv(os.path.join(args.save, "T-1_master.csv"), index=False)

    if "T-2" in FIGURES_TO_MAKE:
        print(f"Making Table 2")

        dfs = {
            "isic": pd.read_csv(os.path.join(PATHS["isic"]["plot"], "df.csv")),
            "pddb": pd.read_csv(os.path.join(PATHS["pddb"]["plot"]["pooled"], "df.csv")),
            "malaria": pd.read_csv(os.path.join(PATHS["malaria"]["plot"], "df.csv")),
        }

        # Columns of dfs: ood_detector,replicate,dist,metric,value
        for dataset in dfs.keys():
            dfs[dataset]["dataset"] = dataset

        dfs["all"] = pd.concat([dfs["isic"], dfs["pddb"], dfs["malaria"]], axis=0)
        dfs["all"].reset_index(drop=True, inplace=True)

        # Make a dataframe of (num_ood_detectors, num_datasets), where each entry
        # is the mean ID - OOD of that ood detector on that dataset
        DATASETS = ["isic", "pddb", "malaria", "all"]
        METRICS = ["ID_AUROC_mean", "ID_AUROC_sd", "OOD_AUROC_mean", "OOD_AUROC_sd", "AUROC_diff_mean", "ID_vs_OOD_prop", "ood_rate_mean", "ood_rate_sd"]
        columns = list()
        for dataset in DATASETS:
            for metric in METRICS:
                columns.append(f"{dataset}::{metric}")
        
        table = pd.DataFrame(index = dfs["all"]["ood_detector"].unique(), columns = columns)
        for ood_detector in tqdm.tqdm(table.index, desc="Make table 2"):
            for dataset in DATASETS:
                for metric in METRICS:
                    col = f"{dataset}::{metric}"
                    df = dfs[dataset]
                    if metric == "ID_AUROC_mean":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "ID") & (df["replicate"] == "main"), "value"].mean()
                    elif metric == "ID_AUROC_sd":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "ID") & (df["replicate"] != "main"), "value"].std()
                    elif metric == "OOD_AUROC_mean":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "OOD") & (df["replicate"] == "main"), "value"].mean()
                    elif metric == "OOD_AUROC_sd":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "OOD") & (df["replicate"] != "main"), "value"].std()
                    elif metric == "AUROC_diff_mean":
                        id_values = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "ID") & (df["replicate"] == "main"), "value"].dropna()
                        ood_values = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "OOD") & (df["replicate"] == "main"), "value"].dropna()
                        val = id_values.mean() - ood_values.mean()
                    elif metric == "ID_vs_OOD_prop":
                        id_values = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "ID") & (df["replicate"] != "main"), "value"].reset_index(drop=True)
                        ood_values = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "AUROC") & (df["dist"] == "OOD") & (df["replicate"] != "main"), "value"].reset_index(drop=True)

                        res = np.where(id_values.isna() | ood_values.isna(), np.nan, id_values > ood_values)
                        res = res[~np.isnan(res)]
                        
                        val = res.mean()
                    elif metric == "ood_rate_mean":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "ood_rate") & (df["dist"] == "ID") & (df["replicate"] == "main"), "value"].mean()
                    elif metric == "ood_rate_sd":
                        val = df.loc[(df["ood_detector"] == ood_detector) & (df["metric"] == "ood_rate") & (df["dist"] == "ID") & (df["replicate"] != "main"), "value"].std()
                    else:
                        raise ValueError(f"Unknown metric {metric}")
                
                    table.loc[ood_detector, col] = val
        
        table.to_csv(os.path.join(args.save, "T-2_oodperformance.csv"), index=True)
    
    if "metrics" in FIGURES_TO_MAKE:
        # General metrics
        table = pd.read_csv(os.path.join(args.save, "T-2_oodperformance.csv"), index_col=0)

        DATASETS = ["isic", "pddb", "malaria"]

        # Compare ID_AUROC_sd to OOD_AUROC_sd
        metrics = {dataset: dict() for dataset in DATASETS}
        for dataset in DATASETS:
            print(dataset)
            print(table[f"{dataset}::ID_AUROC_sd"].mean(), table[f"{dataset}::OOD_AUROC_sd"].mean())

            metrics[dataset]["ID_AUROC_sd"] = table[f"{dataset}::ID_AUROC_sd"].mean()
            metrics[dataset]["OOD_AUROC_sd"] = table[f"{dataset}::OOD_AUROC_sd"].mean()
        
        # Show mean of ID AUROC sd for all datasets
        df = pd.DataFrame(metrics).T
        print(df)
        print(df["ID_AUROC_sd"].mean(), df["OOD_AUROC_sd"].mean())

    print("Done!")

