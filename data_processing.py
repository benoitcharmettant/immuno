from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from utils.image import show_box, crop_patch

path = 'C:/Users/b_charmettant/Desktop/projet_preliminaire/images'
ls_image = [join(path, im) for im in listdir(path)]
im = plt.imread(ls_image[0])[:, :, :3]

coord_patch = (0, 70)
shape_patch = (im.shape[1]//2, int(0.80*im.shape[0]))

show_box(im, coord_patch, shape_patch)
cropped_image = crop_patch(im, coord_patch, shape_patch)
plt.imshow(cropped_image)










