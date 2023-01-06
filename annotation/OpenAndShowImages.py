import cv2
from matplotlib import pyplot as plt

from FileDivisionAssignment import FileDivisionAssignment

import os
import numpy as np
from pycocotools.coco import COCO

TAG = "(coco_processing)"


class OpenAndShowImages:
    def __init__(self, path_images,
                 path_save_annotations,
                 user,
                 load_from_index=False,
                 id_to_load=0):
        self.images_path = path_images
        self.path_save_annotations = path_save_annotations
        self.user = user

        self.load_from_index = load_from_index
        self.__divisor = FileDivisionAssignment(path_images, 10, 2)

        annFile = os.path.join(self.images_path, f'Annotations/annotated_functional_test3_fixed.json')
        print(TAG, f'Annotation file: {annFile}')

        # Load annotations
        self.coco = COCO(annFile)

        # Load image IDs
        self.annIds = self.__divisor.do_split(self.user)
        if load_from_index:
            idx = np.where(self.annIds == id_to_load)[0][0]
            self.annIds = self.annIds[idx:]

        print(TAG, f'Number of images: {len(self.annIds)}')

    def __save_images(self, save_images):
        if save_images:
            for imgId in self.imgIds:
                # save the image
                img = self.coco.loadImgs(imgId)[0]
                file_name = img['file_name']
                img_path = os.path.join(self.images_path, "Images", file_name)
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                cv2.imwrite(os.path.join(self.images_path, "Images", "annotated", img_name), img)

    def __show_image(self, annId, index):
        imgId = self.coco.loadAnns(int(annId))[0]["image_id"]
        img = self.coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        img_path = os.path.join(self.images_path, "Images", file_name)
        img_show = cv2.imread(img_path)

        plt.figure(1, figsize=(12, 8))

        # the left side of the image, show the image and the annotation
        plt.subplot(1, 2, 1)
        plt.text(0, -150, 'CLOSE this window and ENTER the category number,', fontsize=13)
        plt.text(0, -50, 'scratch = 1, dent = 2, rim = 3, other = 4', fontsize=13)
        plt.text(3000, 1800, f'you still have {len(self.annIds) - index} images', fontsize=14)
        plt.axis('off')
        plt.imshow(img_show)
        anns = self.coco.loadAnns(int(annId))
        self.coco.showAnns(anns)

        # the right side of the image, show the detailed annotation
        pos = anns[0]['segmentation'][0]
        img_seg = img_show[pos[1]:pos[7], pos[0]:pos[2], :]

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(img_seg)
        plt.show()

        return anns, imgId

    def start_annotation(self):
        try:
            if not self.load_from_index:
                with open(self.path_save_annotations, "w", encoding='utf-8') as f:
                    f.write("ID_of_Image,ID,Label\n")

            for index, annId in enumerate(self.annIds):
                # get the image
                anns, imgId = self.__show_image(annId, index)

                # get new label to txt file
                label = int(input(f"Please input the label for {annId}: "))
                with open(self.path_save_annotations, "a") as f:
                    f.write(str(imgId) + "," + str(anns[0]['id']) + ',' + str(label) + "\n")
        except KeyboardInterrupt:
            print(f"\n{imgId}\n")
