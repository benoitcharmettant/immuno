from dataset.protocol import Protocol
from viewer.protocol_viewer import Viewer


def main():
    # meta_data_path = "C:/Users/b_charmettant/Desktop/Données_immunothérapies/MK1454/recup_donnees_mk.xlsx"
    # prot_path = "C:/Users/b_charmettant/data/immuno/MK1454"
    # mk = Protocol(prot_path, meta_data_path, "MK")

    meta_data_path = "C:/Users/b_charmettant/Desktop/Données_immunothérapies/LYTIX/recup_donnees_benoit.xlsx"
    prot_path = "C:/Users/b_charmettant/data/immuno/LYTIX"
    lytix = Protocol(prot_path, meta_data_path, "LYTIX")

    protocols = [lytix]
    viewer = Viewer(protocols)

    viewer.start()


if __name__ == '__main__':
    main()
