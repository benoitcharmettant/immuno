from functools import partial
from os import remove
from os.path import join, exists
from tkinter import Tk, Menu, Frame, Canvas, NW, Label, Button
from tkinter.filedialog import askdirectory

from PIL import Image, ImageTk
from numpy import savetxt

from experiment_manager.inspect_results import get_results_image, get_ls_colors
from utils.tools import mkdir, date_to_str
from utils.image import draw_line, get_meta_path, get_ls_patch_coord, get_scale, draw_patches


class Image_Viewer(Frame):
    def __init__(self, viewer):
        Frame.__init__(self, viewer.master)
        self.viewer = viewer
        self.current_image = None
        self.image_canvas = None
        self.label = None
        self.remove_button = None
        self.tk_image = None

        self.width = self.viewer.screen_w
        self.height = self.viewer.screen_h

        self.scale_pos_1 = None
        self.scale_pos_2 = None

    def display_image(self, image_array):
        if self.image_canvas is not None:
            self.image_canvas.pack_forget()
        if self.label is not None:
            self.label.pack_forget()
        if self.remove_button is not None:
            self.remove_button.pack_forget()

        self.label = Label(self.viewer.master, text=self.viewer.get_label_info())

        self.image_canvas = Canvas(self.viewer.master,
                                   height=image_array.shape[0],
                                   width=image_array.shape[1],
                                   cursor='target')

        self.image_canvas.pack()
        self.image_canvas.bind("<Button-1>", self.create_patch)
        self.image_canvas.bind("<Button-3>", self.set_scale)

        image_array = (image_array * 255).astype('uint8')

        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(image_array))

        self.image_canvas.create_image(2, 2, image=self.tk_image, anchor=NW)
        self.label.pack()

        self.remove_button = Button(text="Effacer les patches", command=self.delete_patch_command)
        self.remove_button.pack()

    def create_patch(self, event):
        pos_x = event.x
        pos_y = event.y

        self.viewer.add_patch(self.viewer.current_image, pos_x, pos_y)

        self.viewer.update_display()

    def set_scale(self, event):
        print("hit")
        if self.scale_pos_1 is None:
            self.scale_pos_1 = [event.x, event.y]
        else:
            self.scale_pos_2 = [event.x, event.y]

            self.viewer.set_scale(self.scale_pos_1, self.scale_pos_2)

            self.scale_pos_1, self.scale_pos_2 = None, None

            self.viewer.update_display()

    def delete_patch_command(self):
        image = self.viewer.current_image
        self.viewer.remove_patch_file(image)
        self.viewer.update_display()


class Viewer(object):
    def __init__(self, protocols, patch_size=0.25, **kwargs):
        super(**kwargs)
        self.master = Tk()
        self.master.title('Immuno Annotation')
        self.protocols = protocols

        self.current_exam = None
        self.current_image_nb = 0
        self.current_image = None
        self.base_path = "C:/Users/b_charmettant/data/immuno"
        self.patch_size = patch_size  # in centimeter

        self.allowed_subset = None  # Results for which the colors will be displayed
        self.results_dir = None
        self.result_display_mode = "l1_error"

        # Adapt to screen

        self.screen_w, self.screen_h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+0+0' % (self.screen_w - 20, self.screen_h - 70))
        # print("- Screen_size : {}*{}".format(self.screen_w, self.screen_h))

        # Build viewer

        self.image_viewer = Image_Viewer(self)
        self.master.bind("<Down>", partial(self.change_image_number, 1))
        self.master.bind("<Up>", partial(self.change_image_number, -1))
        self.master.bind("<Right>", partial(self.change_exam, 1))
        self.master.bind("<Left>", partial(self.change_exam, -1))

        self.menubar = Menu_Viewer(self)
        self.master['menu'] = self.menubar

    def start(self):
        self.master.mainloop()

    def change_image_number(self, delta, event):
        if 0 <= self.current_image_nb + delta < len(self.current_exam):
            self.current_image_nb += delta

        self.update_display()

    def change_exam(self, delta, event):
        self.menubar.change_exam_by_delta(delta)

    def update_exam(self, exam, image_nb=0):
        self.current_exam = exam

        assert 0 <= image_nb < len(exam)

        self.current_image_nb = image_nb

    def update_display(self):
        assert self.current_exam is not None

        self.current_image = self.current_exam[self.current_image_nb]

        image_array = self.current_image['image']

        ls_patches_coord = get_ls_patch_coord(self.current_image, self.base_path)
        scale, scale_coord = get_scale(self.current_image, self.base_path)

        # todo: gérer le cas ou le patient n'a pas de résultat plus proprement
        if self.results_dir is not None:
            results_df = get_results_image(self.current_image, self.results_dir)
            if results_df is not None:
                ls_colors = get_ls_colors(ls_patches_coord, results_df, self.allowed_subset,
                                          mode=self.result_display_mode)
                image_array = draw_patches(image_array, ls_patches_coord, scale, self.patch_size, color_list=ls_colors)
            else:
                image_array = draw_patches(image_array, ls_patches_coord, scale, self.patch_size)
        else:
            image_array = draw_patches(image_array, ls_patches_coord, scale, self.patch_size)

        if scale_coord is not None:
            image_array = draw_line(image_array, scale_coord[0], scale_coord[1])

        self.image_viewer.display_image(image_array)

        self.image_viewer.pack()

    def get_label_info(self):
        return "Protocole : {} | Patient : {} | Examen : {} | Image : {} | Machine : {} | Type : {} | Début tt : {} ".format(
            self.current_image['protocole'],
            self.current_image['patient'],
            date_to_str(self.current_image['date']),
            self.current_image['nom_image'],
            self.current_image['machine'],
            self.current_image['type'],
            date_to_str(self.current_image['debut_tt_patient'])
        )

    def set_scale(self, pos1, pos2):
        directory, file = get_meta_path(self.current_image, self.base_path, 'scale')
        path = join(directory, file)
        mkdir(directory)
        savetxt(path, [pos1, pos2], delimiter=",", fmt='%s')

    def add_patch(self, image, x, y):
        ls_patch = get_ls_patch_coord(image, self.base_path)
        ls_patch.append([x, y])
        self.update_patch_file(image, ls_patch)

    def update_patch_file(self, image, ls_patch):
        file_dir, file_name = get_meta_path(image, self.base_path, 'patch')
        mkdir(file_dir)
        savetxt(join(file_dir, file_name), ls_patch, delimiter=",", fmt='%s')

    def remove_patch_file(self, image):
        directory, file = get_meta_path(image, self.base_path, 'patch')
        path = join(directory, file)
        if exists(path):
            remove(path)


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

        self.results_menu = Menu(self, tearoff=False)
        self.add_cascade(label=' Results ', menu=self.results_menu)

        self.results_menu.add_command(label="Dossier resultats",
                                      command=self.change_results_dir)

        self.result_showing_subset = Menu(self.results_menu, tearoff=False)
        self.results_menu.add_cascade(label='Montrer subset... ', menu=self.result_showing_subset)

        self.result_showing_subset.add_command(label="Train",
                                               command=partial(self.change_subset, ['train']))
        self.result_showing_subset.add_command(label="Validation",
                                               command=partial(self.change_subset, ['val']))


        self.result_showing_mode = Menu(self.results_menu, tearoff=False)
        self.results_menu.add_cascade(label='Montrer info... ', menu=self.result_showing_mode)

        self.result_showing_mode.add_command(label="Prédictions",
                                               command=partial(self.change_mode, 'pred'))
        self.result_showing_mode.add_command(label="L1 erreur",
                                               command=partial(self.change_mode, 'l1_error'))
        self.result_showing_mode.add_command(label="L2 erreur",
                                             command=partial(self.change_mode, 'l2_error'))

        self.results_menu.add_command(label="Effacer",
                                      command=self.clear_result_dir)

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
            self.exam_menu.add_command(label="Exam {} - {}".format(i, date_to_str(exam_key)),
                                       command=partial(self.change_exam, exam))

        self.change_exam(ls_exams[list(ls_exams)[0]])

    def change_exam(self, exam):
        self.current_exam = exam

        self.viewer.update_exam(self.current_exam)

        self.viewer.update_display()

    def change_exam_by_delta(self, delta):
        dict_exams = self.current_tumor.get_exams()
        exams_keys = list(dict_exams)

        current_pos = exams_keys.index(self.current_exam[0]['date'])

        if 0 <= current_pos + delta < len(exams_keys):
            current_pos += delta

        self.change_exam(dict_exams[exams_keys[current_pos]])

    def change_results_dir(self):
        path = askdirectory()
        self.viewer.results_dir = path
        self.viewer.allowed_subset = ['train']
        self.viewer.result_display_mode = "l1_error"
        self.viewer.update_display()

    def clear_result_dir(self):
        self.viewer.results_dir = None
        self.viewer.allowed_subset = None
        self.viewer.result_display_mode = None
        self.viewer.update_display()

    def change_subset(self, allowed_subset):
        self.viewer.allowed_subset = allowed_subset
        self.viewer.update_display()

    def change_mode(self, mode):
        self.viewer.result_display_mode = mode
        self.viewer.update_display()

def clear_menu(menu):
    nb_children = len(menu._tclCommands)
    menu.delete(0, nb_children)
