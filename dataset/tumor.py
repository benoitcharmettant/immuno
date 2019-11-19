from os.path import basename

from matplotlib.pyplot import imread


class Tumor(object):
    def __init__(self, name, localisation, patient):
        super()
        self.name = name
        self.patient = patient
        self.loc = localisation

        self.injections = []
        self.ls_images = {}

        self.patient.protocol.add_tumor(self)

    def new_image(self, path_image, date, machine):
        image = imread(path_image)

        new_image = {'image': image,
                     'date': date,
                     'machine': machine,
                     'path': path_image}

        self.ls_images[basename(path_image)] = new_image

    def add_injection(self, date):
        self.injections.append(date)
