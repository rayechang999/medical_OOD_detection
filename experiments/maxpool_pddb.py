import os
import argparse
import json

import numpy as np
import pandas as pd
import sklearn

import torch

import tqdm

def get_metrics_by_individual(df, model_predictions, y):
    if len(df) == 0:
        return {"accuracy": 0.0, "AUROC": 0.0}

    df = df.copy(deep=True)
    
    # model_predictions is an (N, 2) array, take the softmax in the last dim if needed
    if not torch.all(torch.abs(torch.sum(model_predictions, dim=1) - 1) < 1e-5):
        model_predictions = torch.softmax(model_predictions, dim=1)
    
    # Get the model predictions for each individual; individuals have unique healthCode values in df
    # Get the max over all records for each individual
    df["model_predictions_0"] = model_predictions[:, 0]
    df["model_predictions_1"] = model_predictions[:, 1]
    df["y"] = y

    # Create a new dataframe where each row is an individual (a unique healthCode), where the kept row (in the case of multiple rows in the original df for this individual)
    # is whichever one has the largest model_predictions_1
    df = df.loc[df.groupby("healthCode")["model_predictions_1"].idxmax()]
    df = df[["healthCode", "model_predictions_0", "model_predictions_1", "y"]]

    # Get accuracy and AUROC
    accuracy = np.mean(df["y"] == np.argmax(df[["model_predictions_0", "model_predictions_1"]].values, axis=1))
    auroc = sklearn.metrics.roc_auc_score(df["y"], df["model_predictions_1"])

    if len(np.unique(y)) == 1:
        auroc = float("nan")

    return {"accuracy": float(accuracy), "AUROC": float(auroc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str,
                        help="Path to load data from, should be the output folder of a benchmark experiment")
    parser.add_argument("--save", type=str,
                        help="Path to save results to")
    # parser.add_argument("--df", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # Description of this file:
    # This file takes in the output of a benchmark experiment, which contains
    # OOD predictions for each OOD detector as well as the corresponding
    # data and model predictions. The issue is that each observation is a particular
    # time series recording, but we are interested in predicting at the level of individuals.
    # To do this, in the ID subset, we take the model's prediction for that individual
    # as the max over all that individual's ID records; for the OOD subset, we take
    # the model's prediction for that individual as the max over all OOD records.
    # We do the bootstrapping at the record-level, as before.

    datasets = {
        "train": pd.read_csv(os.path.join(args.load, "data", "train.csv"), index_col=0),
        "val": pd.read_csv(os.path.join(args.load, "data", "val.csv"), index_col=0),
        "test": pd.read_csv(os.path.join(args.load, "data", "test.csv"), index_col=0),
    }

    # Load the data (N, m) where N is the number of records and m is the number of features
    MODEL_PATH = os.path.join(args.load, "models", "base")
    model_outputs = {
        "train_predictions": torch.load(os.path.join(MODEL_PATH, "train_predictions.pt")),
        "train_y": torch.load(os.path.join(MODEL_PATH, "train_y.pt")),
        "val_predictions": torch.load(os.path.join(MODEL_PATH, "val_predictions.pt")),
        "val_y": torch.load(os.path.join(MODEL_PATH, "val_y.pt")),
        "test_predictions": torch.load(os.path.join(MODEL_PATH, "test_predictions.pt")),
        "test_y": torch.load(os.path.join(MODEL_PATH, "test_y.pt")),
    }

    OOD_DETECTOR_PATH = os.path.join(args.load, "ood_detectors")
    ood_detector_outputs = dict()
    for ood_detector in sorted(os.listdir(OOD_DETECTOR_PATH)):
        if not os.path.isdir(os.path.join(OOD_DETECTOR_PATH, ood_detector)):
            continue

        ood_detector_outputs[ood_detector] = {
            "id_obs": pd.read_csv(os.path.join(OOD_DETECTOR_PATH, ood_detector, "id_obs.csv"), index_col=0),
            "ood_obs": pd.read_csv(os.path.join(OOD_DETECTOR_PATH, ood_detector, "ood_obs.csv"), index_col=0),
        }
        with open(os.path.join(OOD_DETECTOR_PATH, ood_detector, "id_ood_indices.json"), "r") as f:
            ood_detector_outputs[ood_detector]["id_ood_indices"] = json.load(f)
        
        # Get a boolean mask for the ID/OOD split
        ood_detector_outputs[ood_detector]["test_id_mask"] = np.array(
            [i in ood_detector_outputs[ood_detector]["id_ood_indices"]["id_indices"] for i in range(len(datasets["test"]))]
        )
        ood_detector_outputs[ood_detector]["test_ood_mask"] = np.array(
            [i in ood_detector_outputs[ood_detector]["id_ood_indices"]["ood_indices"] for i in range(len(datasets["test"]))]
        )

        # Check to make sure ID indices and OOD indices are disjoint
        assert np.sum(ood_detector_outputs[ood_detector]["test_id_mask"] & ood_detector_outputs[ood_detector]["test_ood_mask"]) == 0

        # Check to make sure test[id_indices] and test[ood_indices] match id_obs, ood_obs
        assert np.all(datasets["test"].loc[ood_detector_outputs[ood_detector]["test_id_mask"], :].index == ood_detector_outputs[ood_detector]["id_obs"].index)
        assert np.all(datasets["test"].loc[ood_detector_outputs[ood_detector]["test_ood_mask"], :].index == ood_detector_outputs[ood_detector]["ood_obs"].index)
    
    # Get train and val performances (non-bootstrapped)
    train_performance = get_metrics_by_individual(datasets["train"], model_outputs["train_predictions"], model_outputs["train_y"])
    val_performance = get_metrics_by_individual(datasets["val"], model_outputs["val_predictions"], model_outputs["val_y"])

    # Save train and val performances
    with open(os.path.join(args.save, "train_performance.json"), "w") as f:
        json.dump(train_performance, f, indent=2)
    with open(os.path.join(args.save, "val_performance.json"), "w") as f:
        json.dump(val_performance, f, indent=2)

    # Create a directory to save the OOD detector outputs
    OOD_DETECTOR_SAVE_PATH = os.path.join(args.save, "ood_detectors")
    os.makedirs(OOD_DETECTOR_SAVE_PATH, exist_ok=True)

    # Get the performance for each OOD detector
    for ood_detector in tqdm.tqdm(ood_detector_outputs.keys(), desc="Bootstrap OOD detectors"):
        
        performance = dict()

        for replicate in tqdm.tqdm(list(range(1000)) + ["main",], desc=f"Bootstrap {ood_detector}", total=1001):
            replicate = str(replicate)
            performance[replicate] = dict()

            if replicate == "main":
                replicate_inds = np.arange(len(datasets["test"]))
            else:
                replicate_inds = np.random.choice(np.arange(len(datasets["test"])), size=len(datasets["test"]), replace=True)
            
            # Get the model predictions for this replicate
            model_predictions = model_outputs["test_predictions"][replicate_inds]
            y = model_outputs["test_y"][replicate_inds]
            df = datasets["test"].iloc[replicate_inds, :]

            # Get the ID/OOD split
            id_mask = ood_detector_outputs[ood_detector]["test_id_mask"][replicate_inds]
            ood_mask = ood_detector_outputs[ood_detector]["test_ood_mask"][replicate_inds]

            # Get the ID and OOD subsets
            id_subset = df[id_mask]
            ood_subset = df[ood_mask]

            # Get the model predictions for the ID and OOD subsets
            id_model_predictions = model_predictions[id_mask]
            ood_model_predictions = model_predictions[ood_mask]

            # Get the metrics for the ID and OOD subsets
            performance[replicate]["metadata"] = {
                "num_ood": len(ood_subset),
                "num_id": len(id_subset),
            }
            performance[replicate]["ID"] = get_metrics_by_individual(id_subset, id_model_predictions, y[id_mask])
            performance[replicate]["OOD"] = get_metrics_by_individual(ood_subset, ood_model_predictions, y[ood_mask])
            performance[replicate]["all"] = get_metrics_by_individual(df, model_predictions, y)

        # Save the performance for this OOD detector
        os.makedirs(os.path.join(OOD_DETECTOR_SAVE_PATH, ood_detector), exist_ok=True)
        with open(os.path.join(OOD_DETECTOR_SAVE_PATH, ood_detector, "performance.json"), "w") as f:
            json.dump(performance, f, indent=2)
        
    print("Done!")
