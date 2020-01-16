from os.path import join, dirname

from cv2 import resize
from numpy import savetxt
from torch import load, Tensor
import matplotlib.pyplot as plt

from dataset.protocol import Protocol
from utils.image import get_meta_path, get_ls_patch_coord, get_patches, get_dict_split
from utils.tools import mkdir

path_model = '/home/opis/bcharmme/logs/brightness_aug_repeat/convnet_lr1e-05_e2000_bs30_ps0.4_s40_r0.02/best_model.pth'
path_dataset = '/data/bcharmme/immuno/'
print('Loading model...', end=" ")
model = load(path_model)
print('Done !')

patients_mk = ['immuno_{}'.format(i) for i in range(21)]
patients_lt = ['immuno_{}'.format(i) for i in range(21, 41)]

print('Loading data...', end=" ")
mk = Protocol(path_dataset, 'MK1454')
print('Done !')

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

                patch = patch[:, :, :3]
                resized_patch = resize(patch, (model.input_shape[0], model.input_shape[1]))

                tensor_patch = Tensor(resized_patch).to(model.device).reshape((1,
                                                                               3,
                                                                               model.input_shape[0],
                                                                               model.input_shape[1]))

                pred = model(tensor_patch).cpu().item()

                subset = dict_split[f'{coord[0]}_{coord[1]}']
                label = (image['date'] - image['debut_tt_patient']).days > 0
                # format : coord1, coord2, prediction, label(0/1), substet(train/val)
                ls_preds.append([coord[0], coord[1], pred, int(label), subset])

            if len(ls_preds) > 0:

                dir_path, pred_file = get_meta_path(image, path_dataset, 'pred')

                dir_path = join(dirname(path_model), 'predictions', dir_path.split('/')[-1])
                mkdir(dir_path, print=False)
                savetxt(join(dir_path, pred_file), ls_preds, delimiter=",", fmt='%s')
