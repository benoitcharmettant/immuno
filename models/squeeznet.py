from torch.nn import Conv2d
from torchvision.models import squeezenet1_1


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# TODO: Move function in file dedicated to transfer learning networks
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_squeezenet(num_classes):
    model = squeezenet1_1(pretrained=True)
    set_parameter_requires_grad(model, False)

    model.classifier[1] = Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
