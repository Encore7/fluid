from src.data_loader import DataLoader
from src.scripts.helper import clear_folder_contents, clear_metadata


def prepare_dataset(
    config: dict[str, str | int | bool], dataset_folder_path: str
) -> None:
    should_prepare_dataset = config.get("prepare-dataset", None)

    if not should_prepare_dataset:
        return

    print("Clearing metadata...")
    clear_metadata()

    print("Preparing dataset...")

    num_of_clients = int(config.get("num-of-clients", 0))
    dataset_name = str(config.get("dataset-name", None))
    alpha = float(config.get("data-loader-alpha", 0))
    dataset_input_feature = str(config.get("dataset-input-feature", None))
    dataset_target_feature = str(config.get("dataset-target-feature", None))

    clear_folder_contents(dataset_folder_path)

    dataloader = DataLoader(
        dataset_input_feature=dataset_input_feature,
        dataset_target_feature=dataset_target_feature,
        dataset_name=dataset_name,
    )

    dataloader.save_datasets_to_disk(
        num_clients=num_of_clients,
        alpha=alpha,
        dataset_folder_path=dataset_folder_path,
    )
