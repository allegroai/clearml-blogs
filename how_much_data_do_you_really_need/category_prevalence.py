"""
PYTHON VERSION: python3.6

Calculating and visualizing categories_prevalence in a dataset.
The dataset metadeta should be given as a json file in either COCO of BDD format.

Usage:
    1. JSON_PATH=path/to/bdd100k_labels_images_train.json
    category_prevalence.py --dataset-format BDD --json-path JSON_PATH
    2. JSON_PATH=path/to/coco/annotations/instances_train2017.json
    category_prevalence.py --dataset-format COCO --json-path JSON_PATH --wanted-categories person,bike,bird,dog

Requirements:
    - trains
    - numpy
    - seaborn
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import voco_categories, bdd_things_categories
from trains import Task


Task.init(
    project_name="Quantify diminishing Returns",
    task_name="Class Distribution without axis labels",
)


def bdd_class_distribution(json_path: Path) -> dict:
    """
    Create and save a dictionary, with the key being the image name and the value the metadata.
    :param json_path: path to the BDD labels json file.
    :return: A dictionary. Key: Category name. Value: number of appearances.
    """
    category_dict = {}
    with open(json_path, "r") as f:
        entries_list = np.asarray(json.load(f))
        for entry in entries_list:
            for label in entry["labels"]:
                category = label["category"]

                category_dict[category] = category_dict.get(category, 0) + 1
    return category_dict


def coco_id_to_category_name(categories: dict) -> dict:
    """
    Creates a dictionary that gives the category name given its COCO id.
    :param categories: the categories dictionary from COCO's JSON file.
    :return: Dictionary: id -> category_name.
    """
    id_to_name_dict = {entry["id"]: entry["name"] for entry in categories}
    return id_to_name_dict


def coco_class_distribution(json_path: Path) -> dict:
    """
    Counts number of accurences for each class in a dataset metadeta coded
    in COCO-style JSON (COCO's instance JSON file).
    :param json_path: Path to the JSON file contains the metadata.
    :return:  A dictionary. Key: Category name. Value: number of appearances.
    """
    category_dict = {}
    with open(json_path, "r") as f:
        data_dict = json.load(f)
        annotation_list = data_dict["annotations"]
        categories = data_dict["categories"]
        id_to_category_dict = coco_id_to_category_name(categories)
        for entry in annotation_list:
            category = id_to_category_dict[entry["category_id"]]
            category_dict[category] = category_dict.get(category, 0) + 1
    return category_dict


def plot_doughnut(category_dict: dict):
    """
    Plots a doughnut chart of the categories prevalence.
    :param category_dict: category name -> number of occurrence.
    """
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))
    fractions = list(category_dict.values())
    wedges, texts = ax.pie(fractions, wedgeprops=dict(width=0.5), startangle=-40)
    legend_labels = [
        f"{label}: {fraction}" for label, fraction in category_dict.items()
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax.set_title("Class prevalence - BDD")
    plt.savefig(fname="class prevalence", dpi=200)
    plt.show()


###########
## Plots ##
###########


def plot_hist(category_dict: dict):
    """
    Plot labels histogram using seaborn.
    """
    labels = list(category_dict)
    fractions = list(category_dict.values())
    sns.barplot(x=labels, y=fractions)
    plt.show()


def plot_bars_matplotlib(category_dict: dict):
    """
    Plot labels histogram using matplotlib.
    """
    labels = list(category_dict)
    fractions = list(category_dict.values())
    index = np.arange(len(labels))
    plt.bar(index, fractions)
    plt.xlabel("Class", fontsize=5)
    plt.ylabel("Number of Appearances", fontsize=5)
    plt.xticks(index, labels, fontsize=5, rotation=80)
    plt.title("Class Prevalence")
    plt.show()


def plot_bars(category_dict: dict):
    """
    Plot barplots which appears nicely on trains server as plotly object.
    :param category_dict:
    :return:
    """
    labels = list(category_dict)
    fractions = list(category_dict.values())
    plt.bar(labels, fractions)
    plt.xlabel("Class")
    plt.ylabel("Number of Appearances")
    plt.title("Class Prevalence")
    plt.show()


def class_sieve(category_dict: dict, to_keep: set):
    """
    In-place function that leave in 'category_dict'
    only the counting of the classes appearing in 'to_keep'
    :param category_dict: Key: Category name. Value: number of appearances.
    :param to_keep: set of labels to keep in 'category dict'
    """
    return {key: value for key, value in category_dict.items() if key in to_keep}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-format",
        choices=["COCO", "BDD"],
        help="the format of the dataset metadeta",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        help="Path to the metadata, saved in json format. "
        "For example, in the BDD dataset, bdd100k_labels_images_train.json "
        "or bdd100k_labels_images_validation.json"
        "files are possible inputs.",
    )
    parser.add_argument(
        "--wanted-categories",
        help="The categories on which you wish to calculate the statistics,"
        "separated by a comma."
        "If None, all the categories of the dataset will be considered.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_format == "COCO":
        category_count = coco_class_distribution(args.json_path)
        if not args.wanted_categories:
            wanted_categoris = voco_categories
    elif args.dataset_format == "BDD":
        category_count = bdd_class_distribution(args.json_path)
        if not args.wanted_categories:
            wanted_categoris = bdd_things_categories
    if args.wanted_categories:
        wanted_categoris = args.wanted_categories.split(",")
    category_count = class_sieve(category_count, wanted_categoris)
    # plot_bars(category_count)
    plot_doughnut(category_count)


if __name__ == "__main__":
    main()
