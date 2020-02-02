"""
PYTHON VERSION: python3.6

Create subsets of a dataset's metadeta or data. Each subsets contains the smaller predecessor subsets.
For example, all the data within a subset of 1% of the data, presented in the 10%-subset as well.
All the in the data within the 10%-subset is contained within the 20%-subset and so on.

Usage:
    category_prevalence.py --original-json-path  path/to/bdd100k_labels_images_train.json --output-directory path/to/output/directory

Requirements:
    - trains
    - numpy
"""
import json
from argparse import ArgumentParser
from typing import Sequence

import numpy as np
from pathlib import Path


def get_datafile_and_number_of_entries(json_file, dataset_fomat: str):
    """
    Given a read json file and a dataset format, this function
    return the metadata in the usable format and counts how many entries are there in the metadata.
    :param json_file: Metadata content in a python dictionary.
    :param dataset_fomat: format of the dataset metadata
    :return: tuple: (datafile, number of entries in this datafile)
    """
    if dataset_fomat == "BDD":
        datafile = np.asarray(json_file)
        return datafile, len(datafile)
    elif dataset_fomat == "COCO":
        return json_file, len(json_file["images"])


def get_sub_dataset(
    image_array,
    entries_array: np.ndarray,
    fraction: float,
    number_of_entries: int,
    dataset_format: str,
    annotations_array=None,
    data_dict=None,
):
    if dataset_format == "BDD":
        return list(image_array[entries_array[: int(fraction * number_of_entries)]])
    elif dataset_format == "COCO":
        image_entry_list = image_array[
            entries_array[: int(fraction * number_of_entries)]
        ]
        annotation_entry_list = annotations_array[
            entries_array[: int(fraction * number_of_entries)]
        ]
        sub_dataset_dict = {
            "info": data_dict["info"],
            "licenses": data_dict["licenses"],
            "images": list(image_entry_list),
            "annotations": list(annotation_entry_list),
            "categories": data_dict["categories"],
        }
        return sub_dataset_dict


def create_subsets(
    input_json_path: Path,
    output_directory: Path,
    fraction_array: Sequence[float],
    dataset_format: str,
):
    """
    Creates sub sets of BDD metadata.
    :param input_json_path: BDD labels JSON file.
    :param output_directory: Folder to save the BDD metadata sub-sets.
    :param fraction_array: Array contains the sizes of the sub datasets.
    The sizes are brought as fractions of the original dataset.
    """
    with open(input_json_path, "r") as f:
        datafile, number_of_entries = get_datafile_and_number_of_entries(
            json.load(f), dataset_format
        )
    entries_array = np.random.permutation(number_of_entries)
    data_dict = datafile if dataset_format == "COCO" else None
    image_array = (
        np.asarray(data_dict["images"]) if dataset_format == "COCO" else datafile
    )
    annotations_array = (
        np.asarray(data_dict["annotations"]) if dataset_format == "COCO" else None
    )
    for fraction in fraction_array:
        with open(output_directory / f"fraction_of_{fraction}", "w") as outfile:
            sub_dataset = get_sub_dataset(
                image_array=image_array,
                entries_array=entries_array,
                fraction=fraction,
                number_of_entries=number_of_entries,
                dataset_format=dataset_format,
                annotations_array=annotations_array,
                data_dict=data_dict,
            )
            json.dump(obj=sub_dataset, fp=outfile)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-format",
        choices=["COCO", "BDD"],
        help="the format of the dataset metadeta",
    )
    parser.add_argument(
        "--original-json-path",
        help="Path to json file. This file should hold all metadata (or data) instances"
        " as entries in a single Python list",
        type=Path,
    )
    parser.add_argument(
        "--output-directory", type=Path, help="Folder to save the metadata sub-sets."
    )
    parser.add_argument(
        "--fraction-array",
        type=list,
        default=[i / 10 for i in range(1, 11)],
        help="Array contains the sizes of the sub datasets."
        "    The sizes are brought as fractions of the original dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_subsets(
        input_json_path=args.original_json_path,
        output_directory=args.output_directory,
        fraction_array=args.fraction_array,
        dataset_format=args.dataset_format,
    )


if __name__ == "__main__":
    main()
