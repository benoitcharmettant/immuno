from os import listdir
from os.path import join

from pandas import ExcelFile

from dataset.patient import Patient


class Protocol(object):
    def __init__(self, path_directory, path_metadata):
        super()
        self.dir_path = path_directory
        self.excel_path = path_metadata

        self.patients_names = listdir(join(self.dir_path, 'images'))
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
