# 파일 불러오기

import pandas as pd
import dask.dataframe as ddf
import numpy as np

import random

day_by_month = {'1월':31, '4월':29, '7월':31, '10월':30}

class Get_files():
    def __init__(self, file_path, count):
        self.path = file_path
        self.count = count

    # 파일 생성 관리
    def get_file(self):
        saved_files = {'1월':[], '4월':[], '7월':[], '10월':[]}


    #소통데이터 불러오기
    def get_its_c(self):
        files = list_1 = list_4 = list_7 = list_10 = []

        for i in range(4):
            this_month = (i + 1) + 3 * i

            for i_low in range(self.count):
                day = random.randrange(1, day_by_month[f'{this_month}월'])
                random_path = self.path['소통'] + f'{this_month}/its_c_{this_month}_{day}_m1.csv'
                globals()[f'df_{this_month}'] = ddf.read_csv(random_path)

                globals()[f'list_{this_month}'].append(globals()[f'df_{this_month}'])

            files.append(globals()[f'list_{this_month}'])







