from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from utils.us import split_twin_image

path = 'C:/Users/b_charmettant/Desktop/projet_preliminaire/images_echo'
ls_image = [join(path, im) for im in listdir(path)]
im = plt.imread(ls_image[1])[:, :, :3]



im_1, im_2 = split_twin_image(im)

print(im_1.shape)
print(im_2.shape)










