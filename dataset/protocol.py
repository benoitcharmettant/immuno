from os import listdir
from os.path import join


from pandas import ExcelFile

from dataset.patient import Patient




class Protocol(object):
    def __init__(self, path_directory,  name):
        super()
        self.dir_path = join(path_directory, name)
        self.excel_path = join(self.dir_path, "recup_donnees.xlsx")
        self.root_data_path = path_directory


        assert name in ['MK1454', 'LYTIX']

        self.name = name

        self.patients_names = sorted(listdir(join(self.dir_path, 'images')))

        self._xl = ExcelFile(self.excel_path)

        self.ls_patients = []
        self.ls_tumors = []

        for patient_name in self.patients_names:
            _ = self.new_patient(patient_name)

    def new_patient(self, name):
        new_patient = Patient(name, self)
        self.ls_patients.append(new_patient)
        return new_patient

    def add_tumor(self, tumor):
        self.ls_tumors.append(tumor)

    def get_patient_meta_data(self, patient):
        return self._xl.parse(patient.name, header=None)

    def get_nb_patients(self):
        return len(self.ls_patients)

    def get_nb_tumors(self):
        return len(self.ls_tumors)

    def get_nb_images(self):
        nb_images = 0
        for tumor in self.ls_tumors:
            nb_images += tumor.get_nb_images()

        return nb_images

    def get_tumors_stats(self):
        control = 0
        injected = 0

        for tumor in self.ls_tumors:
            if tumor.is_injected():
                injected += 1
            else:
                control +=1
        return control, injected
