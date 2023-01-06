import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "OpenAndShowImages.py"))

from OpenAndShowImages import OpenAndShowImages

if __name__ == "__main__":
    NAME_INDEX = {
        "xiangyu": 0,
        "jessica": 1,
        "mikhael": 2,
        "eneko": 3,
        "dominik": 4,
        "zhen": 5,
        "xiaoting": 6,
        "jian tian": 7,
        "jian peng": 8,
        "tianyuan": 9
    }

    name = str(input("Introduce your name: "))
    print("Are you going to start from the beginning or you have already annotated\n"
          "some images and you want to start from that image? (yes for starting from the beginning\n"
          "or no in case you start from a given index (in that case, the index is needed))")
    time.sleep(2)
    from_start = str(input("Start from the beginning (yes/no): "))

    if from_start.lower() == "yes":
        index_given = False
    else:
        index_given = True
    if index_given:
        indx = int(input("Introduce the index of the image to start from: "))
    else:
        indx = -1
    user = NAME_INDEX[name.lower()]

    path_images = r"C:\Users\Domin\Desktop\Data"
    path_save_annotations = os.path.join(os.getcwd(), f"labels_missing.txt")
    image_opener = OpenAndShowImages(path_images, path_save_annotations, user, index_given, indx)

    image_opener.start_annotation()
