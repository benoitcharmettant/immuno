from torch.utils.data import Dataset


class Patch_Classifier_Dataset(Dataset):
    def __init__(self, ls_protocols, allowed_patients_by_protocol, patch_size):  # patch size in centimeter

        assert len(ls_protocols) == len(allowed_patients_by_protocol)

        self.ls_protocols = ls_protocols
        self.patch_size = patch_size

        self.patches = []
        self.labels = []

        for i in range(len(self.ls_protocols)):
            self.collect_data(self.ls_protocols[i], allowed_patients_by_protocol[i])

    def __getitem__(self, index):
        return self.patches[index], self.labels[index]

    def __len__(self):
        return len(self.patches)

    def new_patch(self, patch, label):
        self.patches.append(patch)
        self.labels.append(label)

    def collect_data(self, protocol, allowed_patients):
        for patient in protocol.ls_patients:
            if patient.name in allowed_patients:
                debut_tt = patient.get_patient_info()['debut_tt']
                for tumor in patient.ls_tumors.values():
                    for exam in tumor.get_exams().values():
                        label_exam = 0
                        if (exam[0]['date'] - debut_tt).days > 0:
                            label_exam = 1
                        for image in exam:

                            ls_patches_exam = tumor.get_patches(image, self.patch_size)
                            if ls_patches_exam is not []:

                                for patch in ls_patches_exam:
                                    self.new_patch(patch, label_exam)


