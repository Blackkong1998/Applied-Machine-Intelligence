import base64
import os.path
import sqlite3
from sqlite3 import Error
from typing import Union, Optional, Iterable
import os

from PIL.Image import Image
from pydantic.annotated_types import Any
from werkzeug.datastructures import FileStorage


class DBManager:
    """
    Database manager. Used for any interaction with the database. SQL Lite is used.
    """

    def __init__(self,
                 table_name: str,
                 path: Optional[str] = None,
                 db_name: Optional[str] = None,
                 database_full_path: Optional[str] = None):
        """
        Initialize the database manager.

        Parameters:
        path (str): path where the sqlite file is stored
        db_name (str): name of the sqlite file
        table_name (str): name of the table in the sqlite file
        """
        if database_full_path is None:
            self.path = os.path.join(path, db_name)
        else:
            self.path = database_full_path

        if not self.__check_if_db_exists():
            print(f"DB is not existing at given path. Creating database at {self.path}")
        try:
            self.__connection = sqlite3.connect(self.path, check_same_thread=False)
            self.__cursor = self.__connection.cursor()
            print("DB Connection to SQLite successful")
        except Error as e:
            print(f"The error {e} occurred. Unable to open DB")

        self.table_name = table_name
        self.__create_table()

    def execute_query(self, query: str, query_type: str,
                      arguments_for_query: Optional[Iterable[Any]] = None) -> Union[Any, None]:
        """
        Execute a query.

        Parameters:
        query (str): query to be executed
        query_type (str): type of query, 'insert', 'create', 'select' or 'update'

        Returns:
        Any: result of the query or None if no result is returned
        """
        try:
            if arguments_for_query is None:
                res = self.__cursor.execute(query)
            else:
                res = self.__cursor.execute(query, arguments_for_query)
            self.__connection.commit()
            print(f"Query for: {query_type} executed correctly")
            return res

        except Error as e:
            print(f"The error {e} occurred")

    def insert_entry(self, image: Union[str, bytes, Image], label: int, filename: Optional[str] = None) -> None:
        """
        Insert an entry into the database.

        Parameters:
        image (bytes or str): image to be inserted. It can be either a path to the file or the image itself as bytes.
        label (int): label of the image
        """
        if type(image) is str:
            with open(image, 'rb') as file:
                blobData = base64.b64encode(file.read())
        elif type(bytes):
            blobData = base64.b64encode(image)
        else:
            blobData = image

        if filename is not None:
            query = f"INSERT INTO {self.table_name} (image, label, filename) VALUES (?, ?, ?);"
            self.execute_query(query, "insert", arguments_for_query=(blobData, label, filename))

        else:
            query = f"INSERT INTO {self.table_name} (image, label) VALUES (?, ?);"
            self.execute_query(query, "insert", arguments_for_query=(blobData, label))

    def insert_entry_without_label(self, image: FileStorage, filename: str) -> None:
        blobData = base64.b64encode(image.read())
        query = f"INSERT INTO {self.table_name} (image, filename) VALUES (?, ?);"
        self.execute_query(query, "insert", arguments_for_query=(blobData, filename))

    def update_label_entry(self, filename: str, label: int) -> None:
        query = f"""UPDATE {self.table_name} SET label = ? WHERE filename = ?;"""
        self.execute_query(query, "update", arguments_for_query=(label, filename))

    def __check_if_db_exists(self) -> bool:
        """
        Check if the database exists.

        Returns:
        bool: True if the database exists, False otherwise
        """
        try:
            sqlite3.connect(f'file:{self.path}?mode=rw', uri=True)
        except sqlite3.OperationalError:
            return False
        return True

    def __create_table(self):
        """
        Create the table.
        """
        query = f"""CREATE TABLE IF NOT EXISTS main.{self.table_name} (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL, 
        label INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        filename TEXT
        );"""
        self.execute_query(query, "create")

    def __check_if_threshold_surpassed(self, threshold: int = 100) -> bool:
        """
        Check if a given threshold for images not used for training is surpassed.
        Parameters:
        threshold (int): threshold for images not used for training. Default is 8.

        Returns:
        bool: True if the threshold is surpassed, False otherwise
        """
        query = f"""SELECT COUNT(*)
                    FROM {self.table_name}
                    WHERE label is not NULL;
                    """
        number_items_not_for_training = self.execute_query(query, "select").fetchall()[0][0]
        return number_items_not_for_training >= threshold

    def __get_distinct_labels(self):
        query = f"""SELECT DISTINCT label
                    FROM {self.table_name}
                    WHERE label is not NULL;
                    """
        data = self.execute_query(query, "select").fetchall()
        return [value[0]for value in data]

    def __get_number_distict(self):
        number_items_per_label = dict()
        labels = self.__get_distinct_labels()
        for item in labels:
            query = f"""SELECT COUNT(*)
                        FROM {self.table_name}
                        WHERE label = '{item}';
                        """
            number_items_per_label[item] = self.execute_query(query, "select").fetchall()[0][0]
        return number_items_per_label

    def __check_for_imbalance(self, dev_from_mean: float = 0.7) -> bool:
        number_items_per_label = self.__get_number_distict()
        all_items = sum(list(number_items_per_label.values()))
        mean_if_perfectly_balanced = all_items / len(number_items_per_label)

        for value in number_items_per_label.values():
            if value > mean_if_perfectly_balanced:
                ratio = mean_if_perfectly_balanced / value
            else:
                ratio = value / mean_if_perfectly_balanced

            if ratio < dev_from_mean:
                return False
        return True

    def check_if_suitable_for_training(self, threshold: int = 100, dev_from_mean: float = 0.7) -> bool:
        """
        Check if the database is suitable for training.
        Returns:
        bool: True if the database is suitable for training, False otherwise
        """

        if self.__check_for_imbalance(dev_from_mean=dev_from_mean) \
                and self.__check_if_threshold_surpassed(threshold=threshold):
            return True
        else:
            return False

