#관리

import pandas as pd
import numpy as np

from get_files import Get_files
from save_file import Save_file
from data_filter import Data_filter
from combine_data import Combine_data


class Activate(Get_files, Save_file, Data_filter, Combine_data):
    #컨스트럭터
    def __init__(self, q_data, path, file_path, link):
        self.count_data = q_data
        self.path = path
        self.file_path = file_path
        self.link = link

    def activate_model(self):
        getting = Get_files(self.file_path, self.count_data)
        saved_by_getting = getting.get_file()
        filtering = Data_filter(saved_by_getting, self.count_data)
        filtered_data = filtering.controller()
        combined_data = Combine_data(filtered_data, self.count_data, self.link)
        result = combined_data.combine()
        print(result)
        # last_step = Save_file(combined_data, self.path)


