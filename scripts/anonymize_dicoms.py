import sys

sys.path.append("../")

from utils.tools import mkdir
from utils.data import anonymize_batch
from os import listdir
from os.path import join

path = 'C:/Users/b_charmettant/Desktop/Données_immunothérapies/LYTIX/donnees/'
ls_image = []

for fname in listdir(path):
    if fname[-4:] == '.dcm':
        ls_image.append(fname)

mkdir(join(path, 'images'))


ls_dicoms = [join(path, im) for im in ls_image]
ls_targets = [join(path, "images/{}.png".format(path_dcm.split('.')[0])) for path_dcm in ls_image]

anonymize_batch(ls_dicoms, ls_targets)

