from cv2 import resize
from numpy import array, transpose
from PIL import Image

from torch.utils.data import Dataset

from utils.image import get_patches, get_ls_patch_coord, get_dict_split, get_tumor_roi


# TODO: check dataloader for exp_1 and exp_2

class Patch_Classifier_Dataset(Dataset):
    def __init__(self, ls_protocols, patch_size, resize,  # patch size in centimeter
                 transform=None,
                 subset='train',
                 experiment='exp_1',  # type of experiment, described in README.md
                 exclude_patients=[]):

        assert subset in ['train', 'val']
        assert experiment in ['exp_1', 'exp_2', 'exp_3']

        self.ls_protocols = ls_protocols
        self.patch_size = patch_size
        self.resize = resize
        self.transform = transform
        self.subset = subset
        self.experiment = experiment
        self.excluded_patients = exclude_patients

        self.patches = []
        self.labels = []
        self.coord = []

        for protocol in self.ls_protocols:
            if self.experiment in ['exp_1', 'exp_2']:
                self.collect_data_patches(protocol)  # will collect the patches in the peritumoral region
            elif self.experiment in ['exp_3']:
                self.collect_data_tumor(protocol)  # will gather an ROI centered on the tumor

        self.patches = array(self.patches).reshape((-1, 3, self.resize, self.resize))
        self.labels = array(self.labels).reshape(len(self), -1)  # Label must be of shape (nb_patches, nb_classes)

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

    def new_patch(self, patch, labels, coord):
        patch = resize(patch, (self.resize, self.resize))
        patch = patch[:, :, :3]

        label = self.format_labels(labels)

        self.patches.append(patch)
        self.labels.append(label)
        self.coord.append(coord)

    def collect_data_patches(self, protocol):

        for tumor in protocol.ls_tumors:

            if tumor.patient.name not in self.excluded_patients:

                for exam in tumor.get_exams().values():
                    for image in exam:
                        ls_patches = get_patches(image, protocol.root_data_path, self.patch_size)
                        ls_coord = get_ls_patch_coord(image, protocol.root_data_path)
                        dict_split = get_dict_split(image, protocol.root_data_path)

                        labels = self.get_labels(image)

                        for i, patch in enumerate(ls_patches):
                            if dict_split[f'{ls_coord[i][0]}_{ls_coord[i][1]}'] == self.subset:
                                self.new_patch(patch, labels, ls_coord[i])

    def collect_data_tumor(self, protocol):
        for tumor in protocol.ls_tumors:
            # print(tumor.patient.name)
            # print(tumor.name)
            if tumor.patient.name not in self.excluded_patients:
                for exam in tumor.get_exams().values():
                    for image in exam:

                        tumor_roi, coords_roi = get_tumor_roi(image, protocol.root_data_path)
                        labels = self.get_labels(image)

                        if tumor_roi is not None:
                            tumor_roi = array(tumor_roi)
                            # print(tumor_roi.shape)
                            self.new_patch(tumor_roi, labels, coords_roi)

    def format_labels(self, labels):
        if self.experiment == 'exp_1':
            return labels['treatment']
        if self.experiment in ['exp_2', 'exp_3']:
            return [labels['treatment'], labels['injection']]

    def get_labels(self, image):
        label_treatment = int((image['date'] - image['debut_tt_patient']).days > 0)
        label_injection = int(bool(label_treatment) and (image['type'] == 'Injected'))

        labels = {
            'treatment': label_treatment,  # 1 if the treatment has started else 0
            'injection': label_injection  # 1 if the treatment has started and the tumor is injected
            # else 0
        }

        return labels


def get_labels_subset(dataset):
    print(dataset.labels.shape)

    nb_label_1 = sum(dataset.labels[:, 0])

    return len(dataset) - nb_label_1, nb_label_1
