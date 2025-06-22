#관리

import pandas as pd
import numpy as np

from get_files import Get_files
from save_file import Save_file
from data_filter import Data_filter
from combine_data import Combine_data


class Activate(Get_files, Save_file, Data_filter, Combine_data):
    #컨스트럭터
    def __init__(self, q_data, path, file_path):
        self.count_data = q_data
        self.path = path
        self.file_path = file_path

    def activate_model(self):
        print('파일 불러오기 클래스 생성')
        getting = Get_files(self.file_path, self.count_data)
        print('파일 불러오기 시작')
        saved_by_getting = getting.get_file()
        print('파일 불러오기 완료, 변수 설정 클래스 생성')

        for i in range(4):
            tm = 1 + 3 * i
            for v in range(len(saved_by_getting[f'{tm}월'][0])):
                filtering = Data_filter(saved_by_getting, tm, v)
                print('변수 설정 시작')
                filtered_data = filtering.controller()
                print('변수 설정 완료, 데이터 병합 클래스 생성')
                combined_data = Combine_data(filtered_data, tm, v)
                print('데이터 병합 실시(이 작업은 시간이 많이 걸립니다.)')
                result = combined_data.combine()
                print('데이터 병합 완료, 데이터 저장 중')
                last_step = Save_file(result, self.path, tm, getting.c_days[v])
                last_step.save_data()

                del filtering, filtered_data, combined_data, result, last_step

        print()