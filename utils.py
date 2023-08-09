import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import yaml
from box import Box
import numpy as np
from fastai.vision.all import *
import glob

# Rapid prototyping requires flexible data structures, such as dictionaries. 
# However, in Python that means typing a lot of square brackets and quotes. 
# The following trick defines an attribute dictionary that allows us to address keys 
# as if they were attributes:

def openConfig():
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    return Box(args)

def get_label(path: str, dataset_name: str):
    dataset_split = path.split(dataset_name,1)[1]
    label = int(dataset_split.split('/',2)[1])
    return label - 1

def getDataset(dataset_dir: str, dataset_name: str, csv_dir: str, train_size: float, test_size: float, random_seed: int, **kwargs):
    train_path = os.path.join(csv_dir, 'train.csv')
    test_path = os.path.join(csv_dir, 'test.csv')
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)

    file_paths = np.array(glob.glob(dataset_dir + dataset_name + '/*/*.bmp'))
    labels = np.array([get_label(path, dataset_name) for path in file_paths])
    files_train, labels_train, files_test, labels_test = stratifiedSortedSplit(file_paths, labels, train_size, test_size, random_seed)
    
    train_dataset = pd.DataFrame({
        "file_path": files_train,
        "label": labels_train
    })
    test_dataset = pd.DataFrame({
        "file_path": files_test,
        "label": labels_test
    })
    
    train_dataset.to_csv(train_path)   
    test_dataset.to_csv(test_path)   
    
    return train_dataset, test_dataset

def stratifiedSortedSplit(file_paths: np.array, labels: np.array, 
                    train_size: float, test_size: float, random_seed: int):
    """Splits image paths and labels equally for each class, then sorts them"""
    splitter = StratifiedShuffleSplit(n_splits=1, 
                                      train_size=train_size, test_size=test_size, random_state=random_seed)
    train_indices, test_indices = next(splitter.split(file_paths, labels))
    
    files_train, labels_train = file_paths[train_indices], labels[train_indices]
    files_test, labels_test = file_paths[test_indices], labels[test_indices]

    sort_index = np.argsort(labels_train)
    labels_train = labels_train[sort_index]
    files_train = files_train[sort_index]

    sort_index = np.argsort(labels_test)
    labels_test = labels_test[sort_index]
    files_test = files_test[sort_index]

    return files_train, labels_train, files_test, labels_test
