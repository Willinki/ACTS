import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from os import listdir


def load_pickle_data(file):
    with open(file, 'rb') as file_to_be_loaded:
        file_use = pickle.load(file_to_be_loaded)
        return pd.DataFrame(file_use)


def datafile():
    data_location = os.getcwd() + '/Data'
    list_of_data = listdir(data_location)
    print(list_of_data)
    while True:
        try:
            # ASSUMING ALL DATA COMES IN PICKLE FILES
            name_of_datafile = input("Insert the name of the file where your data is saved (with type of file): ")
            if name_of_datafile not in list_of_data:
                raise FileNotFoundError
            full_path_data = Path(data_location + f'/{name_of_datafile}')
            data_for_further_process = load_pickle_data(full_path_data)
        except FileNotFoundError:
            print("File does not exist in data directory. Please save it here or type in name once more.")
            continue
        else:
            break
    return data_for_further_process


def use_fraction_of_data(full_data):
    while True:
        try:
            answer = str(input("Do you wish to use a fraction of specified data with chosen method: (y/n)? ")).lower()
            if answer == 'y':
                answer2 = float(input("Specify fraction of data to be used (e.g. 0.3): "))
                if 0 < answer2 <= 1:
                    full_data = full_data.head(round(len(full_data)*answer2))
                elif answer2 == 0:
                    print("You have chosen to use none of your data. Choose again or quit process. ")
                    raise ValueError
                else:
                    raise ValueError
            elif answer == 'n':
                break
            else:
                raise ValueError
        except ValueError:
            print("Choose to use full dataset or fraction of it. ")
            continue
        else:
            break
    return full_data


def columns_to_use(data_with_columns):
    print("Provided dataset includes values for following categories: ", data_with_columns.columns.tolist())
    column_list = data_with_columns.columns.tolist()
    while True:
        try:
            answer = str(input("What category would you like to use for feature extraction? (Full name) "))
            if answer not in data_with_columns:
                raise NameError
        except NameError:
            print("Please choose one existing category. Provide full name (including "'_'"")
            continue
        else:
            break

    return data_with_columns[answer]


def run_acts():
    # DATA AND LABELS TO USE
    data = datafile()
    data = use_fraction_of_data()
    data = columns_to_use()
    labels = np.load('/Data/labels.npy')  # USE ONLY FOR THESIS_DATA_LABELLED.PKL


if __name__ == '__main__':
    run_acts()
