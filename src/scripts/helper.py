import json
import os
import shutil

metadata_path = "metadata.json"


def clear_folder_contents(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def save_metadata(data: dict):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    except json.JSONDecodeError:
        print(
            f"Warning: Existing metadata file at {metadata_path} is invalid JSON. It will be replaced."
        )
        existing_metadata = {}

    # Merge (new overrides old on conflicts)
    merged_metadata = {**existing_metadata, **data}

    # Save merged metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, indent=4)


def load_metadata():
    existing_metadata = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    except json.JSONDecodeError:
        print(
            f"Warning: Existing metadata file at {metadata_path} is invalid JSON. It will be replaced."
        )

    return existing_metadata


def clear_metadata():
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("{}")


metadata = load_metadata()
