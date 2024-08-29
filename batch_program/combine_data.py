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



    def __init__(self, data, count, link):
        self.origin_data = data
        self.count = count
        self.link = link

    def combine(self):
        print('소통 데이터 편집 시작')
        self.edit_c_data()
        print('돌발 데이터 편집 시작')
        self.edit_e_data()
        print('링크 데이터 편집 시작')
        self.edit_link()
        print('노드 데이터 편집 시작')
        self.edit_node()

        print('데이터 병합 함수 가동')
        self.combine_all_stuff()

        return self.origin_data

    def edit_c_data(self):
        for i in range(4):
            tm = 1 + 3 * i
            c_files = self.origin_data[f'{tm}월'][0]
            for v in range(self.count):
                c_origin_data = c_files[v].compute()
                c_data = c_origin_data[c_origin_data['링크ID'] == self.link]
                del c_origin_data

                # 컬럼 이름 변경
                c_data = c_data.rename(columns={'링크ID':'LINK_ID'})
                c_data = c_data.astype({'LINK_ID':'str'})

                #date컬럼 생성
                c_data['date'] = pd.to_datetime(c_data['생성일'].apply(self.get_big_date) + ' ' + c_data['생성시분'].apply(self.get_time))
                c_data = c_data[pd.Index([c_data.columns[-1]]).append(c_data.columns[:-1])]

                c_files[v] = c_data
            self.origin_data[f'{tm}월'][0] = c_files

    def edit_e_data(self):
        for i in range(4):
            tm = 1 + 3 * i
            e_files = self.origin_data[f'{tm}월'][1]
            for v in range(self.count):
                e_data = e_files[v].compute()

                e_data = e_data.rename(columns={'링크아이디':'LINK_ID'})
                e_data = e_data.astype({'LINK_ID':'str'})

                # 돌발일시 키워드가 없다고 뜸 ㅅㅂ 이걸 어케 고쳐
                # 2024-08-28 이게 해결되네
                print(e_data['돌발일시'])
                e_data['date'] = pd.to_datetime(e_data['돌발일시'].apply(self.remove_s))
                e_data = e_data[pd.Index([e_data.columns[-1]]).append(e_data.columns[:-1])]

                e_data['date'] = pd.to_datetime(e_data['date'].apply(self.custom_round))

                e_files[v] = e_data

            self.origin_data[f'{tm}월'][1] = e_files

    def edit_link(self):
        link_file = self.origin_data['링크'].compute()
        link_file = link_file.astype({'LINK_ID':'str', 'F_NODE':'str'})
        link_file = link_file.rename(columns={'F_NODE':'NODE_ID'})
        self.origin_data['링크'] = link_file

    def edit_node(self):
        node_file = self.origin_data['노드'].compute()
        node_file = node_file.astype({'NODE_ID':'str'})
        self.origin_data['노드'] = node_file

    def combine_all_stuff(self):
        combined_data = []
        for i in range(4):
            tm = 1 + 3 * i

            c_files = self.origin_data[f'{tm}월'][0]
            e_files = self.origin_data[f'{tm}월'][1]
            link_file = self.origin_data['링크']
            node_file = self.origin_data['노드']

            for v in range(self.count):
                e_data = e_files[v]
                c_data = c_files[v]

                #날 별 결합 데이터(기본값: 소통)
                each_combine = c_data

                #c + e
                print('소통, 돌발 병합')
                each_combine = pd.merge(c_data, e_data, on=['date', 'LINK_ID'], how='left')

                # + link
                print('링크 병합')
                each_combine = pd.merge(each_combine, link_file, on='LINK_ID', how='left')

                # + node
                print('노드 병합')
                each_combine = pd.merge(each_combine, node_file, on='NODE_ID', how='left')


