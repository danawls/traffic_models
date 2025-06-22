# 파일 저장 기능
import pandas as pd
import dask.dataframe as ddf

class Save_file():
    def __init__(self, data, path, tm, count):
        self.data = data
        self.path = path
        self.tm = tm
        self.count = count

    def save_data(self):
        to_save = self.data
        to_path = f'{self.path}/{self.tm}/{self.count}.csv'
        to_save.to_csv(to_path, index=False)
