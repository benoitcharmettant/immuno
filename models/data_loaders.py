from cv2 import resize
from numpy import array, transpose, array_equal
from PIL import Image

from torch.utils.data import Dataset


class Patch_Classifier_Dataset(Dataset):
    def __init__(self, ls_protocols, allowed_patients_by_protocol, patch_size, resize,
                 transform=None):  # patch size in centimeter

        assert len(ls_protocols) == len(allowed_patients_by_protocol)

        self.ls_protocols = ls_protocols
        self.patch_size = patch_size
        self.resize = resize
        self.transform = transform

        self.patches = []
        self.labels = []

        for i in range(len(self.ls_protocols)):
            self.collect_data(self.ls_protocols[i], allowed_patients_by_protocol[i])

        self.patches = array(self.patches)
        self.patches = self.patches.reshape((-1, 3, 40, 40))

    def __getitem__(self, index):

        patch = self.patches[index]

        if self.transform:
            patch = transpose(patch, (1, 2, 0)) * 255
            patch = Image.fromarray(patch.astype('uint8'))
            patch_tf = self.transform(patch)

            patch = transpose(array(patch_tf), (2, 0, 1)) / 255

        return patch, self.labels[index]

    def __len__(self):
        return len(self.patches)

    def new_patch(self, patch, label):
        patch = resize(patch, (self.resize, self.resize))
        patch = patch[:, :, :3]
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




def get_labels_subset(dataset, subset):
    nb_label_1 = 0

    for i in subset.indices:
        nb_label_1 += dataset.labels[i]

    nb_label_0 = len(subset.indices) - nb_label_1

    return nb_label_0, nb_label_1
