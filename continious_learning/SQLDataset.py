import io
from typing import Tuple, Dict
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from db_integration.DBManager import DBManager
from model_dev.augmentations import get_transformation
from model_dev.datasets import get_classes
import base64

class SQLDataset(Dataset):
    def __init__(self, database_path: str, table_name: str, annotation_path: str):
        # Load DB Manager
        self.table_name = table_name
        self.db_manager = DBManager(database_full_path=database_path, table_name=self.table_name)
        self.transform = get_transformation()
        self.class_list, _ = get_classes(annotation_path)

    def __len__(self) -> int:
        query = f"""
        SELECT COUNT(*)
        FROM {self.table_name} 
        """
        return self.db_manager.execute_query(query, "select").fetchall()[0][0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        query = f"""
        SELECT image, label
        FROM {self.table_name}
        WHERE id = {index + 1};
        """

        blob_image, label = self.db_manager.execute_query(query, "select").fetchall()[0]
        # Decode the string
        binary_data = base64.b64decode(blob_image)
        image = Image.open(io.BytesIO(binary_data))

        inp = self.transform(image)

        sample = dict()
        sample['input'] = inp
        sample['label'] = self.class_list.index(label)
        sample['class_name'] = label

        return sample


def create_dataloader_sql(
                      database_path: str,
                      table_name: str,
                      batch_size: int,
                      annotation_path: str,
                      shuffle: bool = True) -> Tuple[SQLDataset, DataLoader]:
    dataset = SQLDataset(database_path, table_name, annotation_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, dataloader

def visualize_sample_sql(database_path, annotation_path, batch_size, table_name, number_plot):
    _, dataloader = create_dataloader_sql(database_path, table_name,
                                            batch_size, annotation_path)

    # Take some samples
    sample = next(iter(dataloader))
    print("Input shape: {}".format(sample['input'].shape))

    # Plot images
    fig, ax = plt.subplots(1, number_plot, figsize=(20, 10))
    for i in range(number_plot):
        ax[i].imshow(sample['input'][i].permute(1, 2, 0))
        ax[i].set_title(sample['class_name'][i])
    plt.show()
