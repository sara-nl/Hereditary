import os
import argparse
import sys
from typing import List, Tuple
import SimpleITK
import numpy as np
from omegaconf import OmegaConf
from typing import Optional

import pandas as pd


import torch
from torch.utils.data import DataLoader, Dataset
import wandb

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

from unet import UNet

from torchvision import transforms
from PIL import Image

init_train = ['FeTS2022_00159','FeTS2022_00172','FeTS2022_00187','FeTS2022_00199','FeTS2022_00211','FeTS2022_00221', \
              'FeTS2022_00235','FeTS2022_00243','FeTS2022_00258','FeTS2022_00269','FeTS2022_00282','FeTS2022_00291', \
              'FeTS2022_00300','FeTS2022_00311','FeTS2022_00321','FeTS2022_00332','FeTS2022_00344','FeTS2022_00353', \
              'FeTS2022_00370','FeTS2022_00380','FeTS2022_00391','FeTS2022_00403','FeTS2022_00413','FeTS2022_00425', \
              'FeTS2022_00440','FeTS2022_01000','FeTS2022_01038','FeTS2022_01046','FeTS2022_01054','FeTS2022_01062', \
              'FeTS2022_01070','FeTS2022_01078','FeTS2022_01086','FeTS2022_01094','FeTS2022_01102','FeTS2022_01110', \
              'FeTS2022_01118','FeTS2022_01126','FeTS2022_01134','FeTS2022_01205','FeTS2022_01213','FeTS2022_01221', \
              'FeTS2022_01229','FeTS2022_01237','FeTS2022_01245','FeTS2022_01253','FeTS2022_01261','FeTS2022_01269', \
              'FeTS2022_01277','FeTS2022_01293','FeTS2022_01307','FeTS2022_01315','FeTS2022_01323','FeTS2022_01331', \
              'FeTS2022_01339','FeTS2022_01347','FeTS2022_01355','FeTS2022_01363','FeTS2022_01371','FeTS2022_01379', \
              'FeTS2022_01387','FeTS2022_01395','FeTS2022_01403']

feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
label_tag = 'Label'

# using numerical header names
numeric_header_names = {'T1': 1, 
                        'T2': 2, 
                        'FLAIR': 3, 
                        'T1CE': 4, 
                        'Label': 5}

# used to map from numbers back to keys
numeric_header_name_to_key = {value: key for key, value in numeric_header_names.items()}

# column names for dataframe used to create csv

# 0 is for the subject name, 1-4 for modes and 5 for label (as above)
train_val_headers = [0, 1, 2, 3, 4, 5]
val_headers = [0, 1, 2, 3, 4]



def get_all_subdirs(folder_name: str) -> List[str]:
    """
    Get all the subdirectories in the specified folder.

    Args:
        folder_name (str): The path of the folder to search for subdirectories.

    Returns:
        list: A list of full paths to the subdirectories in the specified folder.
    """
    # Get all the (sub)directories in the specified folder
    subdirs = [os.path.join(folder_name, f.name) for f in os.scandir(folder_name) if f.is_dir()]
    return subdirs


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Tuple[str, str]: 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to the config file", default="fets.yaml")
    parser.add_argument("--training_folder", help="Training folder name", default=None)
    parser.add_argument("--validation_folder", help="Validation folder name", default=None)
    parser.add_argument("--partition_id", help="Partition ID", default=1, type=int)
    opt, unknown = parser.parse_known_args()
    return opt, unknown


def get_data_folder_name(argconfig: argparse.Namespace) -> str:
    """
    Get the data folder name from either a config file or user input.

    Args:
        config (Namespace): The path of the config file.

    Returns:
        str: The data folder name.
    """
    config = OmegaConf.create()
    try:
        config = OmegaConf.load(argconfig.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {argconfig.config_file}")
        config = OmegaConf.create(vars(argconfig))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if not argconfig.training_folder:
            if not config.training_folder:   
                config.training_folder = input("Enter the training folder name: ")
        else:
            config.training_folder = argconfig.training_folder
        
        if not argconfig.validation_folder:
            if not config.validation_folder:
                config.validation_folder = input("Enter the validation folder name: ")
        else:
            config.validation_folder = argconfig.validation_folder

        if not argconfig.partition_id:
            if not config.partition_id:
                config.partition_id = input("Enter the partition id: ")
        else:
            config.partition_id = argconfig.partition_id

        config.config_file = argconfig.config_file
        OmegaConf.save(config, config.config_file)  
    return config


def train_val_split(subdirs, percent_train, shuffle=True):
    
    if percent_train < 0 or percent_train >=1:
        raise ValueError('Percent train must be >= 0 and < 1.')

    # record original length of subdirs    
    n_subdirs = len(subdirs)
    if n_subdirs == 0:
        raise ValueError('An empty list was provided to split.')
    
    train_subdirs = [subdir for subdir in subdirs if subdir in init_train]
    # limit subdirs to those that do not lie in init_train
    subdirs = [subdir for subdir in subdirs if subdir not in init_train]

    assert len(subdirs) + len(train_subdirs) == n_subdirs

    if shuffle:
        np.random.shuffle(subdirs)
        
    cutpoint = int(n_subdirs * percent_train) - len(train_subdirs)
    
    train_subdirs = train_subdirs + subdirs[:cutpoint]
    val_subdirs = subdirs[cutpoint:]

    if shuffle:
        np.random.shuffle(train_subdirs)
    
    return train_subdirs, val_subdirs


def get_appropriate_file_paths_from_subject_dir(subdir: str, include_labels: bool) -> dict:
    """
    Get the appropriate file paths from a subject directory.

    Args:
        subdir (str): The path of the subject directory.
        include_labels (bool): Whether to include the labels in the output.

    Returns:
        dict: A dictionary containing the file paths.
    """
    inner_dict = {}
    inner_dict['T1'] = os.path.join(subdir, f'{subdir.split(os.sep)[-1]}_t1.nii.gz')
    inner_dict['T1CE'] = os.path.join(subdir, f'{subdir.split(os.sep)[-1]}_t1ce.nii.gz')
    inner_dict['T2'] = os.path.join(subdir, f'{subdir.split(os.sep)[-1]}_t2.nii.gz')
    inner_dict['FLAIR'] = os.path.join(subdir, f'{subdir.split(os.sep)[-1]}_flair.nii.gz')
    if include_labels:
        inner_dict['Label'] = os.path.join(subdir, f'{subdir.split(os.sep)[-1]}_seg.nii.gz')
    return inner_dict


def paths_dict_to_dataframe(paths_dict, train_val_headers, numeric_header_name_to_key):
    
    # intitialize columns
    columns = {header: [] for header in train_val_headers}
    columns['TrainOrVal'] = [] 
    columns['Partition_ID'] = []
    for inst_name, inst_paths_dict in paths_dict.items():
        for key_to_fpath in inst_paths_dict['train']:
            columns['Partition_ID'].append(inst_name)
            columns['TrainOrVal'].append('train')
            for header in train_val_headers:
                if header == 0:
                    # grabbing the the data subfolder name as the subject id
                    columns[header].append(key_to_fpath['Subject_ID'])
                else:
                    columns[header].append(key_to_fpath[numeric_header_name_to_key[header]])
    
    return pd.DataFrame(columns, dtype=str)


def construct_validation_dataframe(paths_dict, val_headers, numeric_header_name_to_key):
    
    # intitialize columns
    columns = {str(header): [] for header in val_headers}
    columns['TrainOrVal'] = [] 
    columns['Partition_ID'] = []
    
    for inst_name, inst_paths_dict in paths_dict.items():

        for key_to_fpath in inst_paths_dict['val']:
            columns['Partition_ID'].append(inst_name)
            columns['TrainOrVal'].append('val')
            for header in val_headers:
                if header == 0:
                    # grabbing the the data subfolder name as the subject id
                    columns[str(header)].append(key_to_fpath['Subject_ID'])
                else:
                    columns[str(header)].append(key_to_fpath[numeric_header_name_to_key[header]])

    df = pd.DataFrame(columns, dtype=str)
    df = df.drop(columns=['TrainOrVal','Partition_ID'])
    df = df.rename(columns={'0': 'SubjectID', '1': 'Channel_0', 
                       '2': 'Channel_1', '3': 'Channel_2', 
                       '4': 'Channel_3'})
    return df


def get_data_from_csv(split_subdirs_path: str, train_csv_path: str, validation_csv_path: str, percent_train: float) -> pd.DataFrame:
    """
    Get data from a CSV file.

    Args:
        csv_path (str): The path of the CSV file to read.

    Returns:
        pd.DataFrame: The data read from the CSV file.
    """
    split_subdirs = pd.read_csv(split_subdirs_path, dtype=str)
    
    if not set(['Partition_ID', 'Subject_ID']).issubset(set(split_subdirs.columns)):
        raise ValueError("The provided csv at {} must at minimum contain the columns 'Partition_ID' and 'Subject_ID', but the columns are: {}".format(split_subdirs_path, list(split_subdirs.columns)))
    
    # sanity check that all subdirs provided in the dataframe are unique
    if not split_subdirs['Subject_ID'].is_unique:
        raise ValueError("Repeated references to the same data subdir were found in the 'Subject_ID' column of {}".format(split_subdirs_path))
    
    inst_names = list(split_subdirs['Partition_ID'].unique())
    
    paths_dict = {inst_name: {'train': [], 'val': []} for inst_name in inst_names}
    for inst_name in inst_names:
        subdirs = list(split_subdirs[split_subdirs['Partition_ID']==inst_name]['Subject_ID'])
        train_subdirs, val_subdirs = train_val_split(subdirs=subdirs, percent_train=percent_train)
        
        for subdir in train_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(conf.training_folder, subdir), include_labels=True)
            inner_dict['Subject_ID'] = subdir
            paths_dict[inst_name]['train'].append(inner_dict)
        
        for subdir in val_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(conf.validation_folder, subdir), include_labels=True)
            inner_dict['Subject_ID'] = subdir
            paths_dict[inst_name]['val'].append(inner_dict)



    # now construct the dataframe and save it as a csv
    df1 =  paths_dict_to_dataframe(paths_dict=paths_dict, 
                                      train_val_headers=train_val_headers, 
                                      numeric_header_name_to_key=numeric_header_name_to_key)
    
    df1.to_csv(train_csv_path, index=False)
    df2 =  construct_validation_dataframe(paths_dict=paths_dict, 
                                             val_headers=val_headers, 
                                             numeric_header_name_to_key=numeric_header_name_to_key)
    # return df
    df2.to_csv(validation_csv_path, index=False)
    return df1, df2


class FetsDataProvider(Dataset):
    def __init__(self, index: pd.DataFrame, mode: str, label_tag: str, ):
        self.data = index
        # self.csv = pd.read_csv(csv_path)
        self.mode = mode
        self.label_tag = label_tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Get the label
        label = SimpleITK.GetArrayViewFromImage(SimpleITK.ReadImage(row[5]))
        # Get the actual nii gz contents
        image_data = SimpleITK.GetArrayViewFromImage(SimpleITK.ReadImage(row[int(self.mode)]))
        
        return [image_data, label]
        # return {
        #     "image": image_data,
        #     "label": label,
        #     "mode": self.mode,
        #     "subject_id": row[0],
        #     "partition_id": row['Partition_ID'],      
        # }


# class FetsDataLoader(DataLoader):
#     def __init__(self, data_provider: FetsDataProvider, batch_size: int, shuffle: bool):
#         super().__init__(data_provider, batch_size=batch_size, shuffle=shuffle)
#         self.data_provider = data_provider

#     def __len__(self):
#         return len(self.data_provider)
    
#     def __getitem__(self, idx):
#         return self.data_provider[idx]
    
    
def simple_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def dice_loss(pred, target, multiclass=False):
    smooth = 1.
    if multiclass:
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    else:
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice



def preprocess_image(image):
    # m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=m, std=s),
    ])
    return preprocess(image)


if __name__ == "__main__":
    known, unknown = parse_arguments()
    print(f'Discarded unknown arguments: {unknown}')

    conf = get_data_folder_name(known)
    # Get subdirectories
    training_index = get_all_subdirs(conf.training_folder)
    validation_index = get_all_subdirs(conf.validation_folder)
    print(f'Training : {len(training_index)}, Validation: {len(validation_index)}')