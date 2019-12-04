from os.path import join
from os import listdir

from utils.data import anonymize_patient
from utils.tools import mkdir, my_print


def main():
    SRC_PATH = "C:/Users/b_charmettant/Desktop/Données_immunothérapies/LYTIX/donnees"
    TARGET_PATH = "C:/Users/b_charmettant/data/immuno/LYTIX"

    my_print(SRC_PATH)

    mkdir(TARGET_PATH)
    mkdir(join(TARGET_PATH, 'images'))

    ls_patient_data = listdir(SRC_PATH)

    for patient in sorted(ls_patient_data):

        my_print("Processing patient : {}".format(patient))
        patient_dir = join(SRC_PATH, patient)

        ls_dicoms = listdir(patient_dir)

        ls_source = [join(patient_dir, name) for name in ls_dicoms]
        ls_targets = [join(TARGET_PATH, 'images', patient, "{}.png".format(name.split('.')[0])) for name in ls_dicoms]

        mkdir(join(TARGET_PATH, 'images', patient))
        anonymize_patient(ls_source, ls_targets)


if __name__ == '__main__':
    main()
