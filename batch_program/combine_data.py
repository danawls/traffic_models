# 데이터 통합 엔진
import dask.dataframe as ddf
import pandas as pd

# 데이터 병합 순선
# 1. 소통 데이터 데이트 컬럼 생성
# 2. 돌발 데이터 데이트 컬럼 생성(반올림 함수)
# 3. 둘 결합
# 4. 링크 데이터 링크 아이디 개명 후(영어로 통일) 결합(위도 경도는 중점으로.)
# 5. 노드 아이디도 결합
# 6. 날씨 데이터와 날씨 지점 데이터 결합
# 7. 기존 결합데이터에서 가장 유클리디안 거리로 가까운 지점명 컬럼 생성
# 8. 결합 날씨 데이터와 결합
# 9. 혼잡빈도 데이터 결합
# 10. 그외 잡 변수 결합

class Combine_data():
    #컨스트럭터 전까진 도구들

    def get_big_date(v):
        a = str(v)
        return a[:4] + '-' + a[4:6] + '-' + a[6:8]

    def get_time(v):
        a = str(v).rjust(4, '0')
        return a[:2] + ':' + a[2:]

    def remove_s(v):
        a = str(v)
        return a[:-2] + '00'

    def custom_round(a):
        s = str(a)
        number = int(s[14:16])

        if number >= 53:
            return s
        else:
            remainder = number % 10  # 1의 자리 숫자를 얻기 위해 10으로 나눈 나머지로 한다.
        result = 0
        if remainder in [0, 1, 2]:
            result = number - remainder  # 0, 1, 2는 0으로 반올림
        elif remainder in [3, 4, 5, 6, 7]:
            result = number - remainder + 5  # 3, 4, 5, 6, 7은 5로 반올림
        else:
            result = number - remainder + 10  # 8, 9는 0으로 반올림

        return s[:14] + str(result) + s[16:]



    def __init__(self, data, count):
        self.origin_data = data
        self.count = count

    def create_date_at_communication(self):
        for i in range(4):
            tm = 1 + 3 * i
            c_files = self.origin_data[f'{tm}월'][0]
            for v in range(self.count):
                # 컬럼 이름 변경
                c_files[v] = c_files[v].rename(columns={'링크ID':'LINK_ID'})
                c_files[v] = c_files[v].astype({'LINK_ID':'str'})

                #date컬럼 생성
                c_files[v]['date'] = pd.to_datetime(c_files[v]['생성일'].apply(self.get_big_date) + ' ' + c_files[v]['생성시분'].apply(self.get_time))
                c_files[v] = c_files[v][pd.Index([c_files[v].columns[-1]]).append(c_files[v].columns[:-1])]
            self.origin_data[f'{tm}월'][0] = c_files

    def edit_e_data(self):
        for i in range(4):
            tm = 1 + 3 * i
            e_files = self.origin_data[f'{tm}월'][1]
            for v in range(self.count):
                e_files[v] = e_files[v].rename(columns={'링크아이디':'LINK_ID'})
                e_files[v] = e_files[v].astype({'LINK_ID':'str'})

                e_files[v]['date'] = pd.to_datetime(e_files[v]['돌발일시'].apply(self.remove_s))
                e_files[v] = e_files[v][pd.Index([e_files[v].columns[-1]]).append(e_files[v].columns[:-1])]

                e_files[v]['date'] = pd.to_datetime(e_files[v]['date'].apply(self.custom_round))

            self.origin_data[f'{tm}월'] = e_files


    def combine_all_stuff(self):
        #1. c + e
        combined_data = []
        for i in range(4):
            tm = 1 + 3 * i

            c_files = self.origin_data[f'{tm}월'][0]
            e_files = self.origin_data[f'{tm}월'][1]

            for v in range(self.count):
                e_data = e_files[v]
                c_data = c_files[v]

                
