from dataset.protocol import Protocol
from viewer.protocol_viewer import Viewer


def main():

    dataset_path = "C:/Users/b_charmettant/data/immuno"
    mk = Protocol(dataset_path, "MK1454")
    lytix = Protocol(dataset_path, "LYTIX")

    protocols = [mk, lytix]
    viewer = Viewer(protocols, patch_size=0.4)

    viewer.start()


if __name__ == '__main__':
    main()
