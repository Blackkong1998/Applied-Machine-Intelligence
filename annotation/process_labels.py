from json_add_annotations import JSON_label_file
import pandas as pd
from pathlib import Path
import numpy as np
import json


def update_json(files_dir, join_files=True):
    if join_files:
        files = sorted((files_dir / "annotation" / "labels_files").glob("labels_*.txt"))
        all_labels = pd.DataFrame()

        for file in files:
            temp_df = pd.read_csv(file, sep=",", header=0)
            all_labels = pd.concat([temp_df, all_labels], ignore_index=True)

        all_labels.drop_duplicates(subset=['ID_of_Image', 'ID'], inplace=True)
        corrected_label = pd.read_csv(files_dir / "annotation" / "labels_files" /
                                      "labels_corrected.txt", sep=",", header=0)
        labels = pd.concat([all_labels, corrected_label], ignore_index=True)

        labels.to_csv("labels_complete.txt")

    # Replace with path to annotated_functional_test3_fixed.json
    json_path = "..\Data\Annotations\annotated_functional_test3_fixed.json"
    label_path = files_dir / "annotation" / "labels_files" / "labels_complete.txt"
    single_label_file = True
    processed_json_path = files_dir / "new_dataset" / "Annotations" / "updated_annotation.json"

    json_obj = JSON_label_file(json_path, label_path, single_label_file, processed_json_path)
    json_obj.execute_all_tasks()


def check_updated_json(files_dir):
    txt_labels = pd.read_csv(files_dir / "annotation" / "labels_files" / "labels_complete.txt",
                             sep=",", header=0)

    json_file = open(files_dir / "new_dataset" / "Annotations" / "updated_annotation.json")
    json_file = json.load(json_file)

    ann_ids = []
    img_ids = []
    cat_ids = []

    for id in range(len(json_file["annotations"])):
        ann_ids.append(json_file["annotations"][id]["id"])
        img_ids.append(json_file["annotations"][id]["image_id"])
        cat_ids.append(json_file["annotations"][id]["category_id"])

    json_labels = pd.DataFrame({"ID_of_Image": img_ids, "ID": ann_ids, "Label": cat_ids})

    json_labels.sort_values(by=["ID_of_Image", "ID"], inplace=True, ignore_index=True)
    txt_labels.sort_values(by=["ID_of_Image", "ID"], inplace=True, ignore_index=True)
    
    try:
        assert (txt_labels.compare(json_labels).empty)
        print("The json file was correctly updated")
    except (AssertionError, ValueError):
        print("The json file was NOT correctly updated")


if __name__ == "__main__":
    files_dir = Path(__file__).resolve().parent.parent
    update_json(files_dir, join = False)
    check_updated_json(files_dir)