from math import isnan
from os import listdir
from os.path import join

from dataset.tumor import Tumor
from utils.tools import my_print


class Patient(object):
    def __init__(self, name, protocol):
        super()
        self.name = name
        self.protocol = protocol
        self.dir = join(self.protocol.dir_path, 'images', self.name)

        self.meta_data = self.protocol.get_patient_meta_data(self)

        self.patient_info = self.get_patient_info()
        self.ls_images_name = listdir(self.dir)

        self.ls_tumors = {}

        # Extracting data from images
        for image in self.ls_images_name:
            tumor = self.get_tumor(image.split('_')[1])
            date = self.get_image_info(image, 'date')
            machine = self.get_image_info(image, 'machine')
            dim_cible = self.get_dim_cible(image)
            tumor.new_image(join(self.dir, image), date, machine, dim_cible)

        # parsing injection information

        self.parse_injections()

    def get_tumor(self, name):
        if name in list(self.ls_tumors):
            return self.ls_tumors[name]
        return self.new_tumor(name)

    def new_tumor(self, name):
        localisation = self.get_tumor_localisation(name)
        new_tumor = Tumor(name, localisation, self)
        self.ls_tumors[name] = new_tumor
        return new_tumor

    def get_patient_info(self):
        header = self.meta_data.values[:2]
        patient_info = {}
        for i, info in enumerate(header[0]):
            if type(info) == str:
                patient_info[info] = header[1][i]
        return patient_info


    def get_dim_cible(self, image):
        dim = self.get_image_info(image, 'dim_cible')
        if dim.replace(" ", "") == "":
            my_print("Couldn't find the cible's dimension for {}".format(image))

        return dim

    def get_image_info(self, image_name, info):

        # set of info that can be parsed, for dim_cible, the id of
        # the cible is automatically deduced from the image name.
        assert info in ['date', 'cycle', 'machine', 'mise_en_service', "dim_cible"]

        num_exam = int(image_name.split('_')[0][2:])

        if info == "dim_cible":
            info = "dim_cible_{}".format(int(image_name.split('_')[1][5:]))

        start_i, header = self.find_section_meta_data('exam', return_header=True)
        stop_i = self.find_section_meta_data('injection') - 1

        for i in range(start_i, stop_i):
            if not isnan(self.meta_data.values[i][0]):
                if int(self.meta_data.values[i][0]) == num_exam:
                    info_index = header[info]
                    return self.meta_data.values[i][info_index]

        raise IndexError("exam {} data wasn't found in meta data \n"
                         "(patient : {} - image name : {} - info : {})".format(num_exam, self.name, image_name, info))

    def get_tumor_localisation(self, name):
        start_i = self.find_section_meta_data("cible")
        stop_i = self.find_section_meta_data("exam") - 1
        for i in range(start_i, stop_i):
            if str(self.meta_data.values[i][0]).replace('_', '') == name:
                return self.meta_data.values[i][1]
        raise IndexError("{} localisation wasn't found in meta data \n(Patient : {})".format(name, self.name))

    def find_section_meta_data(self, name, return_header=False, default_val=None):
        """
        returns the index of the beginning of a section in the patient's meta data (excluding header)
        :param return_header: If true return a description of the header (name, index dictionary)
        :param name: string, name of the section (content of the first cell (column A))
        :return: int, index of the beginning of the section, header excluded
        """
        name = name.lower()
        assert name in ['patient', 'cible', 'exam', 'injection', 'biopsie']

        found = False
        for i, line in enumerate(self.meta_data.values):
            if name == str(line[0]).lower():
                found = True
                break

        if found:
            if return_header:
                header = {}
                for j, el in enumerate(line):
                    if not type(el) is float:
                        header[el.replace(' ', '')] = j
                return i + 1, header  # dictionary of the index keyed by the name of the section
            return i + 1
        else:

            if default_val is None:
                my_print("Section {} wasn't found".format(name))
                raise NameError("Section {} wasn't found in meta data".format(name))
            else:
                return default_val

    def parse_injections(self):

        start_i, header = self.find_section_meta_data("injection", return_header=True)
        stop_i = self.find_section_meta_data("biopsie", default_val=len(self.meta_data.values) + 1) - 1

        for i in range(start_i, stop_i):

            if not isnan(self.meta_data.values[i][0]):
                cibles = str(self.meta_data.values[i][header['cible']]).replace(' ', '')
                if not cibles == 'nan':
                    date = self.meta_data.values[i][header['date']]

                    cibles = cibles.split(',')

                    for c in cibles:
                        c = c.replace('_', '')
                        if c in self.ls_tumors.keys():
                            self.ls_tumors[c].add_injection(date)
                        else:
                            print(self.ls_tumors)
                            raise IndexError(
                                '{} wasn\'t found in the patient\'s tumors \n(Patient : {})'.format(c, self.name))
