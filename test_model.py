from os.path import join

from cv2 import resize
from numpy import savetxt
from torch import load
import matplotlib.pyplot as plt

from dataset.protocol import Protocol
from utils.image import get_meta_path, get_ls_patch_coord, get_patches, get_dict_split
from utils.tools import mkdir

path_model = None
path_dataset = 'C:/Users/b_charmettant/data/immuno/'
model = load(path_model)

patients_mk = ['immuno_{}'.format(i) for i in [3, 6, 7, 10, 16]]


mk = Protocol(path_dataset, 'MK1454')

for tumor in mk.ls_tumors:
    if tumor.patient.name in patients_mk:
        for name, image in tumor.ls_images.items():
            dir_path, file = get_meta_path(image, path_dataset, 'patch')
            file_path = join(dir_path, file)

            ls_coord = get_ls_patch_coord(image, path_dataset)
            ls_patch = get_patches(image, path_dataset, 0.4)
            dict_split = get_dict_split(image, path_dataset)

            ls_preds = []

            for i, patch in enumerate(ls_patch):
                coord = ls_coord[i]
                if dict_split[f'{coord[0]}_{coord[1]}'] == 'train':
                    resized_patch = resize(patch, (model.input_shape[0], model.input_shape[1]))
                    pred = model(resized_patch)

                    ls_preds.append([coord[0], coord[1], pred])

            dir_path, pred_file = get_meta_path(image, path_dataset, 'pred')
            mkdir(dir_path)

            savetxt(join(dir_path, pred_file), ls_preds, delimiter=",", fmt='%s')




