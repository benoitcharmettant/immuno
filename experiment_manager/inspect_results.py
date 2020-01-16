import csv
from os import listdir
from os.path import join

from matplotlib.pyplot import plot, title, show, subplots
from pandas import DataFrame, concat
from numpy import array, int32, square, abs, mean
from sklearn import metrics

patients_mk = ['immuno_{}'.format(i) for i in range(21)]
patients_lt = ['immuno_{}'.format(i) for i in range(21, 41)]


def get_meta_data(path_to_csv, type_array=int32):
    with open(path_to_csv, newline='') as csv_file:
        data = array(list(csv.reader(csv_file))).astype(type_array)
    return data


def get_results_table(root_path):
    # todo: gérer le cas ou les résultats ne sont pas disponibles pour un patient
    ls_patients = listdir(root_path)

    df_main = DataFrame(
        columns=['pos_0', 'pos_1', 'pred', 'label', 'subset', 'protocole', 'patient', 'cible', 'nom_image'])

    for patient_dir in ls_patients:
        for file in listdir(join(root_path, patient_dir)):
            data = get_meta_data(join(root_path, patient_dir, file), type_array=str)
            new_df = DataFrame(data, columns=['pos_0', 'pos_1', 'pred', 'label', 'subset'])
            new_df['patient'] = patient_dir
            new_df['protocole'] = 'MK1454' if patient_dir in patients_mk else 'LYTIX'
            new_df['cible'] = file.split('_')[2]
            new_df['nom_image'] = file.split('.')[0][5:]

            df_main = concat([df_main, new_df])

    return df_main


def get_results_image(image, root_path):
    df_protocol = get_results_table(root_path)
    df_patient = df_protocol.loc[df_protocol.patient == image['patient']]

    if len(df_patient) == 0:
        return None

    df_exam = df_patient.loc[df_patient.nom_image == image['nom_image']]

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


def compute_results_image(df_image):
    labels = list(df_image['label'].astype(int))
    preds = list(df_image['pred'].astype(float))

    return preds, labels


def compute_results_patient(df_patient):
    patient_name = df_patient['patient'].tolist()[0]

    labels = []
    preds = []

    preds_consensus = []
    labels_consensus = []

    ls_name_images = sorted(list(set(df_patient.nom_image)))

    for name_image in ls_name_images:
        df_image = df_patient.loc[df_patient.nom_image == name_image]
        preds_patient, labels_patient = compute_results_image(df_image)

        labels.append(labels_patient)
        preds.append(preds_patient)

        preds_consensus.append(mean(preds_patient))
        labels_consensus.append(labels_patient[0])

    nb_images = len(preds)

    preds = [l for sublist in preds for l in sublist]
    labels = [l for sublist in labels for l in sublist]

    nb_patches = len(preds)

    fpr_global, tpr_global, thresholds = metrics.roc_curve(labels, preds)
    auc_global = metrics.roc_auc_score(labels, preds)

    fpr_consensus, tpr_consensus, thresholds = metrics.roc_curve(labels_consensus, preds_consensus)
    auc_consensus = metrics.roc_auc_score(labels_consensus, preds_consensus)

    print(f">>>> {patient_name} ({nb_images} images, {nb_patches} patches)")
    print(">> AUC gobale : {:.3f}".format(auc_global))
    print(">> AUC consensus : {:.3f}".format(auc_consensus))

    fig, axis = subplots(1, 2)

    axis[0].plot([0, 1], [0, 1])
    axis[0].plot(fpr_global, tpr_global, marker='.')

    axis[1].plot([0, 1], [0, 1])
    axis[1].plot(fpr_consensus, tpr_consensus, marker='.')

    show()

    return labels, preds, labels_consensus, preds_consensus


def compute_results_protocol(df_protocol):
    labels = []
    preds = []

    preds_consensus = []
    labels_consensus = []

    ls_name_patients = sorted(list(set(df_protocol.patient)))

    for name_patient in ls_name_patients:
        df_patient = df_protocol.loc[df_protocol.patient == name_patient]
        p_label, p_pred, p_label_consensus, p_pred_consensus = compute_results_patient(df_patient)

        labels.append(p_label)
        preds.append(p_pred)

        preds_consensus.append(p_pred_consensus)
        labels_consensus.append(p_label_consensus)

    preds = [l for sublist in preds for l in sublist]
    labels = [l for sublist in labels for l in sublist]

    preds_consensus = [l for sublist in preds_consensus for l in sublist]
    labels_consensus = [l for sublist in labels_consensus for l in sublist]

    fpr_global, tpr_global, thresholds = metrics.roc_curve(labels, preds)
    auc_global = metrics.roc_auc_score(labels, preds)

    fpr_consensus, tpr_consensus, thresholds = metrics.roc_curve(labels_consensus, preds_consensus)
    auc_consensus = metrics.roc_auc_score(labels_consensus, preds_consensus)

    nb_images, nb_patches = len(labels_consensus), len(labels)

    print(f">>>> Résultats Globaux ({nb_images} images, {nb_patches} patches)")
    print(">> AUC gobale : {:.3f}".format(auc_global))
    print(">> AUC consensus : {:.3f}".format(auc_consensus))

    fig, axis = subplots(1, 2)

    axis[0].plot([0, 1], [0, 1])
    axis[0].plot(fpr_global, tpr_global, marker='.')

    axis[1].plot([0, 1], [0, 1])
    axis[1].plot(fpr_consensus, tpr_consensus, marker='.')

    show()
