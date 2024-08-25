#관리

import pandas as pd
import numpy as np

from get_files import Get_files

class Activate(Get_files):
    #컨스트럭터
    def __init__(self, q_data, path, file_path):
        self.count_data = q_data
        self.path = path
        self.file_path = file_path

    def activate_model(self):
        getting = Get_files(self.file_path, self.count)
