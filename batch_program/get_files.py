# 파일 불러오기

import pandas as pd
import dask.dataframe as ddf
import numpy as np

import random


day_by_month = {'1월':31, '4월':29, '7월':31, '10월':30}

class Get_files:
    def __init__(self, file_path, count):
        self.path = file_path
        self.count = count
        self.c_days = []

    # 파일 생성 관리
    def get_file(self):
        saved_files = {'1월':[], '4월':[], '7월':[], '10월':[]}
        got_c_files = self.get_its_c()
        got_e_files = self.get_its_e()
        got_w_files = self.get_weather()
        got_con_files = self.get_confusion_data()

        # 파일 합쳐서 전체 데이터 만들기
        for i in range(4):
            this_month = 1 + 3 * i
            saved_files[f'{this_month}월'].append(got_c_files[i])
            saved_files[f'{this_month}월'].append(got_e_files[i])
            saved_files[f'{this_month}월'].append(got_w_files[i])
            saved_files[f'{this_month}월'].append(got_con_files[i])

        saved_files['도로'] = ddf.read_csv(self.path['도로'])
        saved_files['다발지역'] = ddf.read_csv(self.path['다발지역'])
        saved_files['카메라'] = ddf.read_csv(self.path['카메라'])
        saved_files['노드'] = ddf.read_csv(self.path['노드링크'] + '/node_m1.csv')
        saved_files['링크'] = ddf.read_csv(self.path['노드링크'] + '/link_m1.csv')
        saved_files['인구'] = ddf.read_csv(self.path['인구'])

        return saved_files

    #소통데이터 불러오기
    def get_its_c(self):
        files = []

        for i in range(4):
            this_month = 1 + 3 * i
            globals()[f'c_{this_month}'] = []
            globals()[f'list_{this_month}'] = []
            for i_low in range(self.count):
                day = random.randrange(1, day_by_month[f'{this_month}월'])
                globals()[f'c_{this_month}'].append(day)
                random_path = self.path['소통'] + f'/{this_month}/its_c_{this_month}_{day}_m1.csv'
                globals()[f'df_{this_month}'] = ddf.read_csv(random_path)

                globals()[f'list_{this_month}'].append(globals()[f'df_{this_month}'])

            files.append(globals()[f'list_{this_month}'])
            self.c_days.append(globals()[f'c_{this_month}'])

        return files

    def get_its_e(self):
        files = list_1 = list_4 = list_7 = list_10 = []

        #its_c 데이터 기준으로 가져오기
        for i in range(4):
            this_month = 1 + 3 * i

            for i_low in self.c_days[i]:
                path = self.path['돌발'] + f'/{this_month}/its_e_{this_month}_{i_low}_m1.csv'
                globals()[f'df_{this_month}'] = ddf.read_csv(path)
                globals()[f'list_{this_month}'].append(globals()[f'df_{this_month}'])

            files.append(globals()[f'list_{this_month}'])

        return files

    def get_weather(self):

        files = []

        for i in range(4):
            this_month = 1 + 3 * i

            df = ddf.read_csv(self.path['기상'] + f'/weather_{this_month}_m1.csv')
            files.append(df)

        return files

    def get_confusion_data(self):
        files = []

        for i in range(4):
            this_month = 1 + 3 * i

            df = ddf.read_csv(self.path['혼잡빈도'] + f'/confusion_{this_month}_m1.csv')
            files.append(df)

        return files
