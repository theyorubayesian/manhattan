#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil

import mlflow
import mlflow.sklearn
import json

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Preprocessing(BaseEstimator, TransformerMixin):
    """
    This class implements the pre-processing steps
    """
    def fit(self, X, y=None):
        # Nothing to do
        return self

    def transform(self, X, y=None):

        # Need to make a copy to avoid changing the original dataset
        X = X.copy()

        # Feature engineering: add the length of the name of the property
        X["name_length"] = X["name"].map(str).apply(len)

        # Drop features we do not want to use
        X.drop(
            ["host_name", "host_id", "name", "last_review", "id", "neighbourhood"],
            axis=1,
            inplace=True,
        )

        # One-hot encode some categorical variables
        X_onehot = pd.get_dummies(
            X,
            columns=["neighbourhood_group", "room_type"],
            prefix=["ng", "rt"],
            drop_first=True,
        )

        return X_onehot


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Download train and validation data
    train_local_path = run.use_artifact(args.train).file()
    val_local_path = run.use_artifact(args.val).file()

    # Read the downloaded files and divide X and y
    X_train = pd.read_csv(train_local_path)
    y_train = X_train.pop("price")  # this removes the column "price" from X_train and puts it into y_train

    X_val = pd.read_csv(val_local_path)
    y_val = X_val.pop("price")

    logger.info("Preparing sklearn pipeline")

    # Create model training pipeline
    sk_pipe = Pipeline([
        ("preprocessor", Preprocessing()),
        ("random_forest", RandomForestRegressor(**rf_config))
    ])

    # Fit pipeline on train data
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Save the sk_pipe pipeline as a mlflow.sklearn model
    mlflow.sklearn.save_model(sk_pipe, "random_forest_dir")

    # Upload to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Export of the RandomForest in the MLFlow sklearn format",
        metadata=rf_config,
    )
    artifact.add_dir("random_forest_dir")
    wandb.log_artifact(artifact)

    logger.info("Uploading plots to W&B")

    # Plot feature importance
    wandb.sklearn.plot_feature_importances(sk_pipe.named_steps['random_forest'], X_val.columns.values.tolist())

    # Log MAE and r2
    wandb.log({"r2": r_squared, "mae": mae})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument("train", type=str, help="Train dataset")

    parser.add_argument("val", type=str, help="Validation dataset")

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON file containing a dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor."
    )

    parser.add_argument("--output_artifact", type=str, help="Name for the output serialized model", required=True)

    args = parser.parse_args()

    go(args)
