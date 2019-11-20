from os import makedirs
from os.path import exists


def my_print(arg, on=True):
    if on:
        print(arg)


def mkdir(path_to_create):
    if not exists(path_to_create):
        makedirs(path_to_create)
        my_print("Creating path: {}".format(path_to_create))
    else:
        my_print("Already exists: {}".format(path_to_create))


def date_to_str(date):
    date_str = "{} / {} / {}".format(date.day, date.month, date.year)
    return date_str
