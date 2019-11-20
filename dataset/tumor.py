from os.path import basename

from matplotlib.pyplot import imread, show, figure, imshow

from utils.tools import date_to_str


class Tumor(object):
    def __init__(self, name, localisation, patient):
        super()
        self.name = name
        self.patient = patient
        self.loc = localisation

        self.ls_injections = []
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
        self.ls_injections.append(date)

    def is_injected(self):
        if len(self.ls_injections) > 0:
            return True
        return False

    def get_nb_images(self):
        return len(self.ls_images)

    def display(self, figsize=None):
        ls_exams = self.get_exams()

        for exam in list(ls_exams.values()):
            nb_image = len(exam)
            dummy = exam[0]
            last_injection = self.get_last_injection(dummy)
            date = date_to_str(dummy['date'])
            fig = figure(figsize=figsize)
            print("Exam {} ({}) - Last injection : {}".format(date, dummy['machine'],
                                                              date_to_str(last_injection)
                                                              if last_injection is not None else "None"))
            for i in range(nb_image):
                ax = fig.add_subplot(1, nb_image, i + 1)
                ax.set_axis_off()
                img_plot = imshow(exam[i]['image'])
                ax.set_title(basename(exam[i]['path']))
            show()

    def get_exams(self):
        exams = {}

        for image in list(self.ls_images.values()):
            if image['date'] in exams:
                exams[image['date']].append(image)
            else:
                exams[image['date']] = [image]

        return exams

    def get_last_injection(self, image):

        image_date = image['date']

        if not self.is_injected():
            return None

        last_injection = None
        for injection in sorted(self.ls_injections):
            if (image_date - injection).days > 0:
                last_injection = injection
        return last_injection
