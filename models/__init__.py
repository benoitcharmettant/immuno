from models.convnet import Conv_Net


def model_manager(name, input_size):
    if name == 'convnet':
        return Conv_Net(input_size)
    return None
