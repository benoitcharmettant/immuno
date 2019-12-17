from random import randint


class Rotate_90(object):
    """
    Rotate in image randomly by 90deg.
    """

    def __call__(self, image_pil):
        return image_pil.rotate(90 * randint(0, 3))
