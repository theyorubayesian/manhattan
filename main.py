import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_path="config.yaml")
def go(config: DictConfig):

    # Make sure we are logged in to Weights & Biases ("wandb")
    # wandb.login(key=config['main']['wandb_api_key'])

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = hydra.utils.get_original_cwd()
        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, config['main']['components_repository'], "get_data"),
                "main",
                # version="asiwaju",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'basic_cleaning'),
                "main",
                parameters={
                    "input_artifact": config['etl']['basic_cleaning_input'],
                    "output_name": config["etl"]["output_name"],
                    "output_type": config["etl"]["output_type"],
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                }
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'data_check'),
                "main",
                parameters={
                    "csv": f"{config['etl']['output_name']}:latest",
                    "ref": f"{config['etl']['output_name']}:reference",
                    "kl_threshold": config["data_check"]['kl_threshold'],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    root_path, config["main"]["components_repository"], "train_val_test_split"
                ),
                "main",
                parameters={
                    "input": f"{config['etl']['output_name']}:latest",
                    "test_size": config["modeling"]["test_size"],
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'train_random_forest'),
                'main',
                parameters={
                    "train": "train_data.csv:latest",
                    "val": "val_data.csv:latest",
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, config['main']['components_repository']),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                }
            )


if __name__ == "__main__":
    go()
