import pydicom
from matplotlib.pyplot import imsave, subplots, show, savefig
import matplotlib.patches as patches
from utils.tools import my_print


def read_dicom(path, image_only=True):
    ds = pydicom.dcmread(path)
    if image_only:
        image = ds.pixel_array
        return image / 255
    return ds


def anonymize_dicom(dicom_image):
    """
    Anonymize a raw dicom image by cropping top 70px.
    :param dicom_image: raw dicom image HxWxC
    :return: anonymize image as an array
    """
    return dicom_image[70:, :, :]


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


def show_box(image, coord, shape, save=None):
    fig, ax = subplots(1)

    # Display the image
    ax.imshow(image)

    rect = patches.Rectangle(coord, shape[0], shape[1], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    if save is None:
        show()
    else:
        savefig(save)


def crop_patch(image, coord, shape):
    patch = image[coord[1]:coord[1] + shape[1], coord[0]: coord[0] + shape[0], :]
    return patch