from os.path import exists

from torch import load

path_model = None
path_dataset = '/data/bcharmme/immuno/'
model = load(path_model)


mk = Protocol(path_dataset, 'MK1454')

for tumor in mk.ls_tumors:
    for name, image in tumors.ls_images.items():
        dir_path, file = get_meta_path(image, path_dataset, 'patch')
        file_path = join(dir_path, file)

        ls_coord = get_ls_patch_coord(image)

        for coord in ls_patch_coord:


