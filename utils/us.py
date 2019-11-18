from utils.image import crop_patch


def split_twin_image(image):
    """
    A twin image is an image on which two us views are visible.
    Split this image in half to get two images one for each view
    :param image: numpy array HxWxC
    :return: 2 numpy arrays H'xW'xC
    """
    coord_patch_1 = (0, 0)
    coord_patch_2 = (image.shape[1] // 2, 0)
    shape_patch = (image.shape[1] // 2, image.shape[0])

    image1 = crop_patch(image, coord_patch_1, shape_patch)
    image2 = crop_patch(image, coord_patch_2, shape_patch)

    assert image1.shape == image2.shape

    return image1, image2
