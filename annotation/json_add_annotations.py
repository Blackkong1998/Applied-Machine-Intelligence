import json
from pathlib import Path
from glob import glob


class JSON_label_file:
    def __init__(self, json_path, label_path, single_label_file=True, save_path="annotated_json.json"):
        self.json_path = json_path
        self.single_label_file = single_label_file
        self.label_path = label_path
        self.json_file = json.load(open(json_path))
        self.save_path = save_path

    def define_categories(self):
        """For coco format, we need to define the categories aka object classes
        """
        scratch_dict = {"id": 1, "name": "scratch"}
        dent_dict = {"id": 2, "name": "dent"}
        rim_dict = {"id": 3, "name": "rim"}
        other_dict = {"id": 4, "name": "other"}

        # In COCO format, categories is a list of dictionaries. The dictionaries are the object classes.
        self.json_file['categories'] = [scratch_dict, dent_dict, rim_dict, other_dict]

    def check_single_label_file(self):
        """If single_label_file is true, the given label path needs to be a path to a file. Otherwise to a directory.

        Raises:
            Exception: Exception is raised if single_label_file is true and the given label_path is not a file.
            Exception: Exception is raised if single_label_file is false and the given label_path is not a directory.
        """
        label_path_obj = Path(self.label_path)
        if self.single_label_file:
            if not label_path_obj.is_file():
                raise Exception('the label_path should point to a SINGLE txt file containing labels!')
        else:
            if not label_path_obj.is_dir():
                raise Exception('the label_path should point to a DIRECTORY containing txt files!')

    def update_categories(self, label_path):
        """Update the categories for the image_ids listed in the label file

        Args:
            label_path (str): path to a txt file containing labels
        """
        open_label_file = open(label_path, 'r')
        lines = open_label_file.read().splitlines()
        for line in lines[1:]:
            try:
                im_id, id_id, cat_id = line.split(",")

                # In COCO format, annotations is a list of dictionaries. The dictionaries contain the image_id and
                # category_id.
                for annotation in self.json_file['annotations']:
                    if annotation['image_id'] == int(im_id):
                        if annotation['id'] == int(id_id):
                            annotation['category_id'] = int(cat_id)
            except:
                if line == '' or line == '\n':
                    print("detected extra line at the end of file")
                else:
                    print(f"cannot parse this line: {line}")
                break

    def update_json_file(self):
        """update json file either with one single label file, or multiple label files in a dir.
        """
        if self.single_label_file:
            self.update_categories(self.label_path)
        else:
            label_paths = Path(self.label_path).glob('*.txt')
            for label_path in label_paths:
                self.update_categories(label_path)

    def save_updated_json(self):
        """save the updated json into a new json file
        """
        with open(self.save_path, 'w') as file:
            json.dump(self.json_file, file)

    def execute_all_tasks(self):
        """execute all necessary steps to update the categories for annotated images.
        """
        self.define_categories()
        self.check_single_label_file()
        self.update_json_file()
        self.save_updated_json()


if __name__ == "__main__":
    # These variables are defined as if all paths are in the current working directory
    # If that is not the case, please adjust accordingly.
    json_path = "annotated_functional_test3_fixed.json"
    label_path = "label_example.txt"
    single_label_file = True
    processed_json_path = "annotated_json.txt"

    json_obj = JSON_label_file(json_path, label_path, single_label_file, processed_json_path)
    json_obj.execute_all_tasks()
