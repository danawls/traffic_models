# 데이터 통합 엔진
import dask.dataframe as ddf
import pandas as pd
import numpy as np

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
# 11. 링크 별로 그룹화, 중복 제거 후 배치로 내보내기

#f node로

class Combine_data():
    #컨스트럭터 전까진 도구들

    def get_big_date(self, v):
        a = str(v)
        return a[:4] + '-' + a[4:6] + '-' + a[6:8]

    def get_time(self, v):
        a = str(v).rjust(4, '0')
        return a[:2] + ':' + a[2:]

    def remove_s(self, v):
        a = str(v)
        return a[:-2] + '00'

    def custom_round(self, a):
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


    def get_distance(self, a, b):
        #유클라디안 거리 계산기
        sq = (a['위도'] - b['위도']) ** 2 + (a['경도'] - b['경도']) ** 2
        return np.sqrt(sq)

    def return_lat(self, a):
        return round(float(a.split()[2][:-1]), 4)

    def return_long(self, a):
        return round(float(a.split()[1][1:]), 4)

    def get_spot(self, a, b):
        b['거리'] = b.apply(self.get_distance, args=(a,), axis=1)
        m_d = max(b['거리'])
        return b['지점'][b['거리'] == m_d].iloc[0]



    def get_nearest_spot(self, to_get_data):
        spot_list = self.origin_data['지점']
        to_get_data['위도'] = to_get_data['geometry'].apply(self.return_lat)
        to_get_data['경도'] = to_get_data['geometry'].apply(self.return_long)
        to_get_data = to_get_data.drop(['geometry'], axis=1)
        to_get_data['지점'] = to_get_data.apply(self.get_spot, args=(spot_list, ), axis=1)

        return to_get_data

    def __init__(self, data, tm, count):
        self.origin_data = data
        self.count = count
        self.tm = tm

    def combine(self):
        print('소통 데이터 편집 시작')
        self.edit_c_data()
        print('돌발 데이터 편집 시작')
        self.edit_e_data()
        print('링크 데이터 편집 시작')
        self.edit_link()
        print('노드 데이터 편집 시작')
        self.edit_node()
        print('날씨 데이터 편집 시작')
        self.edit_weather()
        print('지점 데이터 편집 시작')
        self.edit_spot()

        print('데이터 병합 함수 가동')
        result_data = self.combine_all_stuff()

        return result_data

    def edit_c_data(self):
        c_files = self.origin_data[f'{self.tm}월'][0][self.count]
        c_data = c_files.compute()
        print('소통데이터 불러오기 완료')

        # 컬럼 이름 변경
        c_data = c_data.rename(columns={'링크ID':'LINK_ID'})
        c_data = c_data.astype({'LINK_ID':'str'})

        #date컬럼 생성
        c_data['date'] = pd.to_datetime(c_data['생성일'].apply(self.get_big_date) + ' ' + c_data['생성시분'].apply(self.get_time))
        c_data = c_data[pd.Index([c_data.columns[-1]]).append(c_data.columns[:-1])]

        self.origin_data[f'{self.tm}월'][0][self.count] = c_data

    def edit_e_data(self):
            e_files = self.origin_data[f'{self.tm}월'][1][self.count]
            e_data = e_files.compute()

            e_data = e_data.rename(columns={'링크아이디':'LINK_ID'})
            e_data = e_data.astype({'LINK_ID':'str'})

            # 돌발일시 키워드가 없다고 뜸 ㅅㅂ 이걸 어케 고쳐
            # 2024-08-28 이게 해결되네
            print(e_data['돌발일시'])
            e_data['date'] = pd.to_datetime(e_data['돌발일시'].apply(self.remove_s))
            e_data = e_data[pd.Index([e_data.columns[-1]]).append(e_data.columns[:-1])]

            e_data['date'] = pd.to_datetime(e_data['date'].apply(self.custom_round))

            self.origin_data[f'{self.tm}월'][1][self.count] = e_data

    def edit_link(self):
        link_file = self.origin_data['링크'].compute()
        link_file = link_file.astype({'LINK_ID':'str', 'F_NODE':'str'})
        link_file = link_file.rename(columns={'F_NODE':'NODE_ID'})
        self.origin_data['링크'] = link_file

    def edit_node(self):
        node_file = self.origin_data['노드'].compute()
        node_file = node_file.astype({'NODE_ID':'str'})
        self.origin_data['노드'] = node_file

    def edit_weather(self):
        weather_data = self.origin_data[f'{self.tm}월'][2][self.count].compute()
        self.origin_data[f'{self.tm}월'][2][self.count] = weather_data


    def edit_spot(self):
        spot_data = self.origin_data['지점'].compute()
        self.origin_data['지점'] = spot_data


    def combine_all_stuff(self):
            c_data = self.origin_data[f'{self.tm}월'][0][self.count]
            e_data = self.origin_data[f'{self.tm}월'][1][self.count]
            link_file = self.origin_data['링크']
            node_file = self.origin_data['노드']
            weather_file = self.origin_data[f'{self.tm}월'][2][self.count]
            spot_file = self.origin_data['지점']

            #날씨 + 지점 결합
            weather_spot = pd.merge(weather_file, spot_file, on='지점', how='inner')

            #c + e
            print('소통, 돌발 병합')
            each_combine = pd.merge(c_data, e_data, on=['date', 'LINK_ID'], how='left')

            # + link
            print('링크 병합')
            each_combine = pd.merge(each_combine, link_file, on='LINK_ID', how='left')

            # + node
            print('노드 병합')
            each_combine = pd.merge(each_combine, node_file, on='NODE_ID', how='left')

            # + weather_spot
            # # 첫번째로 가장 가까운 거리의 지점 컬럼 만들기
            each_combine = self.get_nearest_spot(each_combine)
            each_combine = pd.merge(each_combine, weather_spot, on='지점', how='left')

            return each_combine

