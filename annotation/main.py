from pycocotools.coco import COCO
from pathlib import Path

import matplotlib.pyplot as plt
import cv2

import os


# define the coco dataset process function
def coco_dataset_process(read_ann_from_root=True, save_images=False, new_label_mode=False):
    """
    This function processes the coco dataset and saves the images and annotations.
    :param new_label_mode:
    :param read_ann_from_root:
    :param save_images: default False. If true, the images will be saved.
    :return: nothing.
    """
    TAG = "(coco_processing)"
    # Define the path to the dataset
    cocoRoot = r"C:\Users\Domin\Desktop\Data"

    # Path to parent directory
    dir_path = Path(__file__).resolve().parent.parent

    if read_ann_from_root:
        # Load COCO dataset
        annFile = os.path.join(cocoRoot, 'Annotations/annotated_functional_test3_fixed.json')
        print(TAG, f'Annotation file: {annFile}')
    else:
        annFile = "annotated_json.txt"
    # Load annotations
    coco = COCO(annFile)

    # load the ID of the category
    ids = coco.getCatIds('damage')

    # Load image IDs
    imgIds = coco.getImgIds(catIds=ids)
    print(TAG, f'Number of images: {len(imgIds)}')

    annIds = coco.getAnnIds()
    print(TAG, f'Number of annotated images: {len(annIds)}')

    # save the annotated images
    if save_images:
        for index, annId in enumerate(annIds):
            # get the image id
            imgId = coco.loadAnns(annId)[0]["image_id"]
            # save the image segment
            img = coco.loadImgs(imgId)[0]
            file_name = img['file_name']
            img_path = os.path.join(cocoRoot, f'Images/{file_name}')
            img_name = Path(img_path).stem + "_" + str(index) + ".jpeg"
            img = cv2.imread(img_path)
            anns = coco.loadAnns(annId)

            # the right side of the image, show the detailed annotation
            pos = anns[0]['segmentation'][0]
            img_seg = img[pos[1]:pos[7], pos[0]:pos[2], :]
            cv2.imwrite(str(dir_path / "new_dataset" / "Images" / img_name), img_seg)

    # get the new annotation file
    if new_label_mode:
        for index, annId in enumerate(annIds):
            # get the image
            imgId = coco.loadAnns(annId)[0]["image_id"]
            img = coco.loadImgs(imgId)[0]
            file_name = img['file_name']
            img_path = os.path.join(cocoRoot, f'Images/{file_name}')
            img_show = cv2.imread(img_path)

            plt.figure(1, figsize=(12, 8))

            # the left side of the image, show the image and the annotation
            plt.subplot(1, 2, 1)
            plt.text(0, -150, 'CLOSE this window and ENTER the category number,', fontsize=13)
            plt.text(0, -50, 'scratch = 1, dent = 2, rim = 3, other = 4', fontsize=13)
            plt.text(3000, 1800, f'you still have {len(annIds) - index} images', fontsize=14)
            plt.axis('off')
            plt.imshow(img_show)
            anns = coco.loadAnns(annId)
            coco.showAnns(anns)

            # the right side of the image, show the detailed annotation
            pos = anns[0]['segmentation'][0]
            img_seg = img_show[pos[1]:pos[7], pos[0]:pos[2], :]

            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(img_seg)
            plt.show()

            # get new label to txt file
            label = int(input(f"Please input the label for {annId}: "))
            with open(os.path.join("label_example.txt"), "a") as f:
                f.write(str(imgId) + "," + str(anns[0]['id']) + ',' + str(label) + "\n")

    demage_dict = {1: "scratch", 2: "dent", 3: "rim", 4: "other"}

    # show the information of the specific image
    img_index = 8
    img_to_show = imgIds[img_index]
    img_to_show_Info = coco.loadImgs(imgIds)[img_index]
    print(TAG, f'Image {img_to_show}\'s info: {img_to_show_Info}')

    # show the image
    imPath = os.path.join(cocoRoot, 'Images', img_to_show_Info['file_name'])
    im = cv2.imread(imPath)

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im)

    # get the damage area of the image
    annIds = coco.getAnnIds(imgIds=img_to_show_Info['id'])
    anns = coco.loadAnns(annIds)
    print(TAG, f'the annotation of the image: {anns}')
    plt.text(0, -350, f'Damege category: {demage_dict[anns[0]["category_id"]]}', fontsize=12)
    plt.text(0, -200, f'ID: {anns[0]["id"]}', fontsize=12)
    print(TAG, f'The category of the damage is: {anns[0]["category_id"]}')

    coco.showAnns(anns)

    # show the image with the damage area
    pos = anns[0]['segmentation'][0]
    img_seg = im[pos[1]:pos[7], pos[0]:pos[2], :]

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(img_seg)
    plt.show()


if __name__ == '__main__':
    coco_dataset_process(read_ann_from_root=True, save_images=True, new_label_mode=False)
