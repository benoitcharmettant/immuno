from functools import partial
from tkinter import Tk, Menu


class Viewer(object):
    def __init__(self, protocols, **kwargs):
        super(**kwargs)
        self.master = Tk()
        self.master.title('Immuno Annotation')
        self.dic_protocols = protocols

        self.menubar = Menu_Viewer(self.master, self.dic_protocols)
        self.master['menu'] = self.menubar

    def start(self):
        self.master.mainloop()


class Menu_Viewer(Menu):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.master = master

        self.protocols = self.master.dic_protocols
        self.current_protocol = self.protocols.keys()[0]  # key of self.protocols

        self.protocol_menu = Menu(self)
        self.add_cascade(label='Protocoles', menu=self.protocol_menu)

        self.patient_menu = Menu(self)
        self.add_cascade(label='Patients', menu=self.patient_menu)

        self.tumor_menu = Menu(self)
        self.add_cascade(label='Cible', menu=self.tumor_menu)

        for prot in self.protocols.keys():
            self.protocol_menu.add_command(label=self.protocols[prot].name,
                                           command=partial(self.change_protocol, prot))

        self.change_protocol(self.current_protocol)

    def change_protocol(self, protocol):
        self.current_protocol = self.protocols[0]

        #TODO: ne pas oublier de tout effacer avant d'ajouter les commandes pour chaque menu !

    def change_patient(self, patient):
        pass

    def change_cible(self, cible):
        pass
