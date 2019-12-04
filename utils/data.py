from os.path import basename

from matplotlib.pyplot import imsave

from utils.image import read_dicom, anonymize_dicom
from utils.tools import my_print
from utils.us import split_twin_image


def anonymize_batch(ls_src, ls_targets):
    """
    Anonymize a batch of dicom image to png images
    :param ls_src: list of paths of source dicom images
    :param ls_targets: list of paths of target png images
    """

    assert len(ls_src) == len(ls_targets)

    for i, src in enumerate(ls_src):
        raw_image = read_dicom(src)
        anonymize_image = anonymize_dicom(raw_image)

        imsave(ls_targets[i], anonymize_image)
        my_print("Saving {} into {}".format(src, ls_targets[i]))


def anonymize_patient(ls_src, ls_targets):
    """
    Anonymize a batch of dicom image to png images
    :param ls_src: list of paths of source dicom images
    :param ls_targets: list of paths of target png images
    """

    assert len(ls_src) == len(ls_targets)

    unsaved_image = 0

    for i, src in enumerate(ls_src):
        if not src[-4:] == '.dcm':
            my_print("Skipping file {}".format(src))
        else:

            raw_image, dcm = read_dicom(src, image_only=False)

            name_dicom = basename(src).split('.')[0]

            anonymize_image = anonymize_dicom(raw_image)

            if name_dicom.split('_')[-1] == 'ab':
                image_a, image_b = split_twin_image(anonymize_image)
                my_print('## Splitting image {}'.format(src))
                imsave(ls_targets[i].replace('_ab', '_a'), image_a)
                my_print("Saving {} into {}".format(src, ls_targets[i].replace('_ab', '_a')), on=False)

                imsave(ls_targets[i].replace('_ab', '_b'), image_b)
                my_print("Saving {} into {}".format(src, ls_targets[i].replace('_ab', '_b')), on=False)

            elif name_dicom.split('_')[-1] in ['a', 'b', 'c', 'd']:
                imsave(ls_targets[i], anonymize_image)
                my_print("Saving {} into {}".format(src, ls_targets[i]), on=False)
            else:
                unsaved_image += 1
                my_print("## Wrong format for {}".format(name_dicom))

    my_print('Patient formated (unsaved images : {})'.format(unsaved_image))
