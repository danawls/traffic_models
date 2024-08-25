# 파일 불러오기

import pandas as pd
import dask.dataframe as ddf
import numpy as np

class Get_files():
    def __init__(self, file_path):
        self.path = file_path

    def get_file(self):

