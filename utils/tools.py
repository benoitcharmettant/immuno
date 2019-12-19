import csv
from os import makedirs
from os.path import exists

from numpy import int32, array


def my_print(arg, logger=None, on=True):
    if on:
        print(arg)
    if not logger is None:
        logger.write(arg)


def mkdir(path_to_create, print=True):
    if not exists(path_to_create):
        makedirs(path_to_create)
        if print:
            my_print("Creating path: {}".format(path_to_create))
    else:
        if print:
            my_print("Already exists: {}".format(path_to_create))


def date_to_str(date):
    date_str = "{} / {} / {}".format(date.day, date.month, date.year)
    return date_str


def get_meta_data(path_to_csv, type_array=int32):
    with open(path_to_csv, newline='') as csv_file:
        data = array(list(csv.reader(csv_file))).astype(type_array)
        # TODO: généraliser l'utilisation de cette fonction dans les autres fichiers
    return data
