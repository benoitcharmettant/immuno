from datetime import datetime
from os.path import join

from utils.tools import mkdir


class Logger:
    def __init__(self, path_log, arg):
        dir_name = "{}_lr{}_e{}_bs{}_ps{}_s{}_r{}_d{}/".format(arg.model, arg.lr, arg.epoch, arg.batch_size,
                                                               arg.patch_size,
                                                               arg.resize,
                                                               arg.regul,
                                                               arg.dropout)
        path_log = join(path_log, dir_name)
        mkdir(path_log)
        self.root_dir = path_log
        self.log_file = join(self.root_dir, "console.log")

    def write(self, arg, end='\n'):
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        with open(self.log_file, 'a') as f:
            f.write("{}  >  {}{}".format(dt_string, arg, end))
