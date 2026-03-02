import os
import subprocess

import toml

from src.scripts.helper import clear_folder_contents
from src.scripts.prepare_dataset import prepare_dataset


def pre_flwr_run():
    print("Running pre-flwr setup...")
    pyproject_toml_file_path = "pyproject.toml"

    if not os.path.exists(pyproject_toml_file_path):
        raise FileNotFoundError(
            f"File '{pyproject_toml_file_path}' not found in the current directory."
        )

    with open(pyproject_toml_file_path, "r", encoding="utf-8") as file:
        pyproject_data = toml.load(file)

    config = (
        pyproject_data.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
    )

    logs_folder_path = config.get("logs-folder-path", None)
    dataset_folder_path = config.get("dataset-folder-path", None)

    # clearing logs before running the experiment
    print("Clearing logs and models...")
    clear_folder_contents(logs_folder_path)

    # preparing the dataset
    prepare_dataset(config, dataset_folder_path)


if __name__ == "__main__":
    pre_flwr_run()

    subprocess.run(["flwr", "run"], check=True)
