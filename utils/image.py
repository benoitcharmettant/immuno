import csv
from os.path import exists, join

import pydicom
from PIL import Image
from PIL.ImageDraw import Draw
from matplotlib.pyplot import subplots, show, savefig
from numpy import array, copy, int32
import matplotlib.patches as patches


#
# * Variable named "image" refers to a python dictionary, with a
#   format defined in dataset.tumor (includes meta data about the image)
#
# * Variable name "image_array" refers to a numpy array containing only
#   pixels values of an image
#

def read_dicom(path, image_only=True):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array / 255
    if image_only:
        return image
    return image, ds


def anonymize_dicom(image_dicom):
    """
    Anonymize a raw dicom image by cropping top 70px.
    :param image_dicom: raw dicom image HxWxC
    :return: anonymize image as an array
    """

    return image_dicom[70:, :, :]


def show_box(image_array, coord, shape, save=None):
    fig, ax = subplots(1)

    # Display the image
    ax.imshow(image_array)

    rect = patches.Rectangle(coord, shape[0], shape[1], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    if save is None:
        show()
    else:
        savefig(save)


def get_color(value):
    value = int(255*value)
    color = (value, 255 - value, 0)
    return color


def draw_patches(image_array, patch_list, scale, patch_size, color_list=None):
    if scale is None:
        patch_size_pix = 160 * patch_size
    else:
        patch_size_pix = scale * patch_size

    for i, patch_coord in enumerate(patch_list):
        coord = [int(patch_coord[0]) - patch_size_pix // 2, int(patch_coord[1]) - patch_size_pix // 2]
        if color_list is not None:
            image_array = draw_box(image_array, coord, [patch_size_pix, patch_size_pix], color=color_list[i])
        else:
            image_array = draw_box(image_array, coord, [patch_size_pix, patch_size_pix])

    return image_array


def draw_box(image_array, coord, shape, color=None):
    image_array = (image_array * 255).astype('uint8')
    pil_image = Image.fromarray(image_array)
    draw = Draw(pil_image)

    if color is not None:
        color = get_color(color)

    draw.rectangle(((coord[0], coord[1]),
                    (coord[0] + shape[0], coord[1] + shape[1])),
                   outline=color if color is not None else 'blue',
                   width=3 if color is not None else 1)

    image_array = copy(array(pil_image))

    return image_array * 255


def draw_line(image_array, coord_1, coord_2):
    image_array = (image_array * 255).astype('uint8')
    pil_image = Image.fromarray(image_array)

    draw = Draw(pil_image)
    draw.line([(int(coord_1[0]) - 2, int(coord_1[1]) - 2), (int(coord_2[0]) - 2, int(coord_2[1]) - 2)], fill='green',
              width=5)

    image_array = copy(array(pil_image))

    return image_array * 255


def crop_patch(image_array, coord, shape):
    patch = image_array[int(coord[1]):int(coord[1]) + int(shape[1]), int(coord[0]): int(coord[0]) + int(shape[0]), :]
    return patch


def get_scale(image, base_path):
    """
    The scale of an image should have been set manually to 1 cm after the US image scale
    :param base_path: Base directory for meta_data
    :param image: image in the format described in dataset.tumor
    :param get_positions: Return the coordinates of the scale ends instead
    :return: Returns the scale (pixels per cm) saved in the meta scale_*.csv,
             if the scale wasn't measure returns None.
    """
    file_dir, file_name = get_meta_path(image, base_path, 'scale')

    if not exists(join(file_dir, file_name)):
        return None, None

    with open(join(file_dir, file_name), newline='') as csv_file:
        data = array(list(csv.reader(csv_file))).astype(int32)

    return abs(data[0][1] - data[1][1]), [data[0], data[1]]


def get_meta_path(image, base_path, meta):
    dir_path = join(base_path, image['protocole'], 'meta', image['patient'])
    file_name = "{}_{}.csv".format(meta, image['nom_image'])

    return dir_path, file_name


def get_ls_patch_coord(image, base_path):
    file_dir, file_name = get_meta_path(image, base_path, 'patch')

    if not exists(join(file_dir, file_name)):
        return []

    with open(join(file_dir, file_name), newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return data


def get_dict_split(image, base_path):
    file_dir, file_name = get_meta_path(image, base_path, 'split')

    if not exists(join(file_dir, file_name)):
        return []

    with open(join(file_dir, file_name), newline='') as csvfile:
        data = list(csv.reader(csvfile))

    formatted_data = {}

    for l in data:
        formatted_data[f"{l[0]}_{l[1]}"] = l[2]

    return formatted_data


def get_patches(image, base_path, patch_size):
    ls_patch = []

    scale, _ = get_scale(image, base_path)
    patches_coord = get_ls_patch_coord(image, base_path)

    if scale is None:
        patch_size_pix = 160 * patch_size
    else:
        patch_size_pix = scale * patch_size

    for coord in patches_coord:
        coord_patch = [int(coord[0]) - patch_size_pix // 2, int(coord[1]) - patch_size_pix // 2]

        patch = crop_patch(image['image'], coord_patch, [patch_size_pix, patch_size_pix])

        ls_patch.append(patch)

    return ls_patch
