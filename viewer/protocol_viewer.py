import csv
from functools import partial
from os.path import join, exists
from tkinter import Tk, Menu, Frame, Canvas, NW, Label
from PIL import Image, ImageTk
from numpy import savetxt

from utils.tools import mkdir


class Image_Viewer(Frame):
    def __init__(self, viewer):
        Frame.__init__(self, viewer.master)
        self.viewer = viewer
        self.current_image = None
        self.image_canvas = None
        self.label = None
        self.tk_image = None

        self.width = self.viewer.screen_w
        self.height = self.viewer.screen_h

    def display_image(self, image):
        if not self.image_canvas is None:
            self.image_canvas.pack_forget()
        if not self.label is None:
            self.label.pack_forget()

        self.label = Label(self.viewer.master, text="Hello !")

        self.current_image = image

        image_array = self.current_image['image'][:, :, :3]

        self.image_canvas = Canvas(self.viewer.master,
                                   height=image_array.shape[0],
                                   width=image_array.shape[1])

        self.image_canvas.pack()
        self.image_canvas.bind("<Button-1>", self.create_patch)

        image_array = (image_array * 255).astype('uint8')

        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(image_array))

        self.image_canvas.create_image(2, 2, image=self.tk_image, anchor=NW)

        self.label.pack()

    def create_patch(self, event):
        pos_x = event.x
        pos_y = event.y

        self.viewer.add_patch(self.current_image, pos_x, pos_y)

        self.viewer.update_display()


class Viewer(object):
    def __init__(self, protocols, **kwargs):
        super(**kwargs)
        self.master = Tk()
        self.master.title('Immuno Annotation')
        self.protocols = protocols

        self.current_exam = None
        self.current_image_nb = 0
        self.base_path = "C:/Users/b_charmettant/data/immuno"

        # Adapt to screen

        self.screen_w, self.screen_h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+0+0' % (self.screen_w - 20, self.screen_h - 80))
        # print("- Screen_size : {}*{}".format(self.screen_w, self.screen_h))

        # Build viewer

        self.image_viewer = Image_Viewer(self)

        self.menubar = Menu_Viewer(self)
        self.master['menu'] = self.menubar

    def start(self):
        self.master.mainloop()

    def update_exam(self, exam, image_nb=0):
        self.current_exam = exam

        assert 0 <= image_nb < len(exam)

        self.current_image_nb = image_nb

    def update_display(self):
        assert self.current_exam is not None

        image = self.current_exam[self.current_image_nb]

        self.image_viewer.display_image(image)
        self.image_viewer.pack()

    def add_patch(self, image, x, y):
        ls_patch = self.get_ls_patch(image)
        ls_patch.append([x, y])
        self.update_patch(image, ls_patch)

    def update_patch(self, image, ls_patch):
        file_dir, file_name = self.get_meta_path(image)
        mkdir(file_dir)
        savetxt(join(file_dir, file_name), ls_patch, delimiter=",", fmt='%s')

    def get_ls_patch(self, image):
        file_dir, file_name = self.get_meta_path(image)

        if not exists(join(file_dir, file_name)):
            return []

        with open(join(file_dir, file_name), newline='') as csvfile:
            data = list(csv.reader(csvfile))
        return data

    def get_meta_path(self, image):
        dir_path = join(self.base_path, image['protocole'], 'meta', image['patient'])
        file_name = "patch_{}.csv".format(image['nom_image'])

        return dir_path, file_name


class Menu_Viewer(Menu):
    def __init__(self, viewer):
        super().__init__(viewer.master)

        self.viewer = viewer

        self.protocols = self.viewer.protocols

        self.current_protocol = None
        self.current_patient = None
        self.current_tumor = None
        self.current_exam = None

        self.protocol_menu = Menu(self, tearoff=False)
        self.add_cascade(label=' Protocoles ', menu=self.protocol_menu)

        self.patient_menu = Menu(self, tearoff=False)
        self.add_cascade(label=' Patients ', menu=self.patient_menu)

        self.tumor_menu = Menu(self, tearoff=False)
        self.add_cascade(label=' Cibles ', menu=self.tumor_menu)

        self.exam_menu = Menu(self, tearoff=False)
        self.add_cascade(label=' Examens ', menu=self.exam_menu)

        for prot in self.protocols:
            self.protocol_menu.add_command(label=prot.name,
                                           command=partial(self.change_protocol, prot))

        self.change_protocol(self.protocols[0])

    def change_protocol(self, protocol):

        clear_menu(self.patient_menu)

        self.current_protocol = protocol

        for patient in self.current_protocol.ls_patients:
            self.patient_menu.add_command(label=patient.name,
                                          command=partial(self.change_patient, patient))

        self.change_patient(self.current_protocol.ls_patients[0])

    def change_patient(self, patient):

        clear_menu(self.tumor_menu)

        self.current_patient = patient

        for tumor_key in list(patient.ls_tumors):
            tumor = patient.ls_tumors[tumor_key]
            self.tumor_menu.add_command(label=tumor.name,
                                        command=partial(self.change_tumor, tumor))

        self.change_tumor(patient.ls_tumors[list(patient.ls_tumors)[0]])  # set the viewer on the first tumor

    def change_tumor(self, tumor):

        clear_menu(self.exam_menu)

        self.current_tumor = tumor
        ls_exams = tumor.get_exams()
        for i, exam_key in enumerate(list(ls_exams)):
            exam = ls_exams[exam_key]
            self.exam_menu.add_command(label="Exam {} - {}".format(i, exam_key),
                                       command=partial(self.change_exam, exam))

        self.change_exam(ls_exams[list(ls_exams)[0]])

    def change_exam(self, exam):
        self.current_exam = exam

        self.viewer.update_exam(self.current_exam)

        self.viewer.update_display()


def clear_menu(menu):
    nb_childs = len(menu._tclCommands)
    menu.delete(0, nb_childs)
