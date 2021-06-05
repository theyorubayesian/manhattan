#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in Weights & Biases
"""
import argparse
import logging
import os

import pandas as pd
import wandb
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    df = df[df['price'].between(args.min_price, args.max_price)]

    df['last_review'] = pd.to_datetime(df['last_review'])
    df['last_review'].fillna(pd.to_datetime("2010-01-01"), inplace=True)

    df['reviews_per_month'].fillna(0, inplace=True)

    df['name'].fillna('-', inplace=True)
    df['host_name'].fillna('-', inplace=True)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(os.path.join("outputs", args.output_name), index=False)
    logger.info(f"Uploading {args.output_name} to Weights & Biases")
    log_artifact(
        args.output_name,
        args.output_type,
        args.output_description,
        os.path.join("outputs", args.output_name),
        run
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans data")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help='',
        required=True
    )

    parser.add_argument(
        "--output_name",
        type=str,
        help='',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='',
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help='',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help='',
        required=True
    )
    args = parser.parse_args()

    go(args)
