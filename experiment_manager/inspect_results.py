from os import listdir
from os.path import join

from utils.tools import get_meta_data

from pandas import DataFrame


def get_results_table(root_path):
    ls_patients = listdir(root_path)

    df = DataFrame(columns =['pos_0', 'pos_1', 'pred', 'label', 'subset'])

    for patient_dir in ls_patients:
        for file in listdir(join(root_path, patient_dir)):
            data = get_meta_data(join(root_path, patient_dir, file), type=str)

            return data
