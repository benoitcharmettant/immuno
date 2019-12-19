import csv
from os import listdir
from os.path import join

from pandas import DataFrame, concat
from numpy import array, int32, square, abs


def get_meta_data(path_to_csv, type_array=int32):
    with open(path_to_csv, newline='') as csv_file:
        data = array(list(csv.reader(csv_file))).astype(type_array)
    return data


def get_results_table(root_path):
    # todo: gérer le cas ou les résultats ne sont pas disponibles pour un patient
    ls_patients = listdir(root_path)

    df_main = DataFrame(columns=['pos_0', 'pos_1', 'pred', 'label', 'subset', 'patient', 'nom_image'])

    for patient_dir in ls_patients:
        for file in listdir(join(root_path, patient_dir)):
            data = get_meta_data(join(root_path, patient_dir, file), type_array=str)
            new_df = DataFrame(data, columns=['pos_0', 'pos_1', 'pred', 'label', 'subset'])
            new_df['patient'] = patient_dir
            new_df['nom_image'] = file.split('.')[0][5:]

            df_main = concat([df_main, new_df])

    return df_main


def get_results_image(image, root_path):
    df_protocol = get_results_table(root_path)
    df_patient = df_protocol.loc[df_protocol.patient == image['patient']]

    if len(df_patient) == 0:
        return None

    df_exam = df_patient.loc[df_patient.nom_image == image['nom_image']]
    df_exam
    return df_exam[['pos_0', 'pos_1', 'pred', 'label', 'subset']]


def get_ls_colors(ls_coords_patch, results_df, allowed_subsets, mode="pred"):
    assert mode in ['pred', 'l1_error', 'l2_error']

    ls_colors = []

    ls_pos_0 = results_df['pos_0']
    ls_pos_1 = results_df['pos_1']

    # todo: make sure there can't be the same patch twice for an image (same position)
    results_df['pos_id'] = [f'{ls_pos_0[i]}_{ls_pos_1[i]}' for i in range(len(ls_pos_0))]

    for coord in ls_coords_patch:

        patch_data = results_df.loc[results_df.pos_id == f"{coord[0]}_{coord[1]}"]
        subset = patch_data['subset'].iloc[0]
        pred = float(patch_data['pred'].iloc[0])
        label = float(patch_data['label'].iloc[0])

        color = None  # color must be between 0 and 1

        if mode == 'pred':
            color = pred
        elif mode == 'l1_error':
            color = abs(pred - label)
        elif mode == 'l2_error':
            color = square(pred - label)

        if subset in allowed_subsets:
            ls_colors.append(color)
        else:
            ls_colors.append(None)

    return ls_colors


if __name__ == "__main__":
    df = get_results_table("C:/Users/b_charmettant/logs/predictions/")
    df_val = df.loc[df.subset == 'val']
    df_train = df.loc[df.subset == 'train']
