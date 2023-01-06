import os
import json
import numpy as np

NAME_INDEX = {
    "Xiangyu": 0,
    "Jessica": 1,
    "Mikhael": 2,
    "Eneko": 3,
    "Dominik": 4,
    "Zhen": 5,
    "Xiaoting": 6,
    "Jian Tian": 7,
    "Jian Peng": 8,
    "Tianyuan": 9
}


class FileDivisionAssignment:
    np.random.seed(0)

    def __init__(self, image_folder_path: str,
                 number_of_annotators: int = 10,
                 times_annotated_same_image: int = 2):
        path_annotations = os.path.join(image_folder_path, "Annotations", "annotated_functional_test3_fixed.json")
        with open(path_annotations, 'r', encoding='utf-8') as f:
            self.annotation_data = json.loads(f.read())
        self.number_of_annotators = number_of_annotators
        self.times_annotated_same_image = times_annotated_same_image

    def do_split(self, user):
        lst_ids = np.array(self.__get_ids())
        np.random.shuffle(lst_ids)

        splits = np.array_split(lst_ids, self.number_of_annotators)

        if user == len(splits) - 1:
            if user - 4 >= 0:
                return np.concatenate((splits[user], splits[user - 4]), axis=0)
            else:
                return np.concatenate((splits[user], splits[user - 2]), axis=0)
        return np.concatenate((splits[user], splits[user + 1]), axis=0)

    def __get_ids(self):
        return [self.annotation_data["annotations"][i]["id"] for i in range(len(self.annotation_data["annotations"]))]
