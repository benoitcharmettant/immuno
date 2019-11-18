import pydicom
from matplotlib.pyplot import subplots, show, savefig
import matplotlib.patches as patches


def read_dicom(path, image_only=True):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array / 255
    if image_only:
        return image
    return image, ds


def anonymize_dicom(dicom_image):
    """
    Anonymize a raw dicom image by cropping top 70px.
    :param dicom_image: raw dicom image HxWxC
    :return: anonymize image as an array
    """

    return dicom_image[70:, :, :]


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
