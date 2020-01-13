from os import remove
from os.path import join, exists
from random import random

from numpy import savetxt

from dataset.protocol import Protocol
from utils.image import get_meta_path, get_ls_patch_coord, get_dict_split
from utils.tools import mkdir


def random_split(protocol, ratio, overwrite=False):
    """
    Create a random repartition between training and validation data.
    This repartition is saved in the dataset meta folder under the files "split_[image_name].csv"
    :param protocol: Protocol to be split
    :param ratio: Ration of training data over validation
    :param overwrite: Will overwrite a previous distribution
    :return: Nothing.
    """
    print(f"Splitting protocol {protocol.name}...")
    for tumor in protocol.ls_tumors:
        for name, image in tumor.ls_images.items():
            path_dataset = protocol.root_data_path
            ls_coord = get_ls_patch_coord(image, path_dataset)

            ls_split = get_dict_split(image, path_dataset)

            new_split = []
            for coord in ls_coord:
                if overwrite:
                    new_split.append([coord[0],
                                      coord[1],
                                      'train' if random() < ratio else 'val'])
                else:
                    if f"{coord[0]}_{coord[1]}" in ls_split.keys():
                        new_split.append([coord[0],
                                          coord[1],
                                          ls_split[f"{coord[0]}_{coord[1]}"]])
                    else:
                        new_split.append([coord[0],
                                          coord[1],
                                          'train' if random() < ratio else 'val'])

            file_dir, file_name = get_meta_path(image, path_dataset, 'split')

            if not new_split == []:
                mkdir(file_dir)
                savetxt(join(file_dir, file_name), new_split, delimiter=",", fmt='%s')
            else:
                if exists(join(file_dir, file_name)):
                    remove(join(file_dir, file_name))


if __name__ == "__main__":
    data_path = "C:/Users/b_charmettant/data/immuno/"
    MK = Protocol(data_path, "MK1454")
    random_split(mk, 0.9, True)


