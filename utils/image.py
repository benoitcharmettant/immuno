import pydicom
from PIL import Image
from PIL.ImageDraw import Draw
from matplotlib.pyplot import subplots, show, savefig, imshow
from numpy import array, copy
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


def draw_box(image_array, coord, shape):
    image_array = (image_array * 255).astype('uint8')
    pil_image = Image.fromarray(image_array)

    draw = Draw(pil_image)
    draw.rectangle(((coord[0], coord[1]), (coord[0] + shape[0], coord[1] + shape[1])), outline='red')

    image_array = copy(array(pil_image))

    return image_array * 255


def draw_line(image_array, coord_1, coord_2):
    image_array = (image_array * 255).astype('uint8')
    pil_image = Image.fromarray(image_array)



    draw = Draw(pil_image)
    draw.line([(int(coord_1[0])-2, int(coord_1[1])-2), (int(coord_2[0])-2, int(coord_2[1])-2)], fill='green', width=5)

    image_array = copy(array(pil_image))

    return image_array * 255


def crop_patch(image, coord, shape):
    patch = image[coord[1]:coord[1] + shape[1], coord[0]: coord[0] + shape[0], :]
    return patch
