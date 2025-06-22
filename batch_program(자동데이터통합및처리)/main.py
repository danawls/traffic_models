from batch_program.supervise import Activate


file_paths = {'소통':'/Volumes/Expansion/traffic-prediction/data/its-소통',
              '돌발':'/Volumes/Expansion/traffic-prediction/data/its-돌발',
              '기상':'/Volumes/Expansion/traffic-prediction/data/기상청-방재',
              '도로':'/Volumes/Expansion/traffic-prediction/data/도로 데이터/road(info)_m1.csv',
              '다발지역':'/Volumes/Expansion/traffic-prediction/data/전국교통사고다발지역표준데이터/전국교통사고다발지역표준데이터.csv',
              '카메라':'/Volumes/Expansion/traffic-prediction/data/전국무인교통단속카메라표준데이터/전국무인교통단속카메라표준데이터_m1.csv',
              '노드링크':'/Volumes/Expansion/traffic-prediction/data/표준노드링크/data',
              '인구':'/Volumes/Expansion/traffic-prediction/data/행정안전부_지역별(행정동) 성별 연령별 주민등록 인구수/행정안전부_지역별(행정동) 성별 연령별 주민등록 인구수_20240731.csv',
              '혼잡빈도':'/Volumes/Expansion/traffic-prediction/data/혼잡빈도',
              '지점':'/Volumes/Expansion/traffic-prediction/data/other/META_관측지점정보_20240825213356.csv'}

def greeting():
    print('안녕하세요! 교통류 이론과 GRU를 결합한 교통예측모델 개발에 관한 연구에 필요한 데이터 생성기 프로그램을 시작합니다!\n이 프로그램은 연구 진행자인 최우진에 의해 개발되었으며,'
          '연구 이외에 다른 용도로는 사용하지 않습니다.\n')

    user_input = input('데이터를 생성하시겠습니까? Y/N: ')

    if user_input == 'Y':
        get_argument()
    elif user_input == 'N':
        exit()
    else:
        print('올바르지 않은 커맨드입니다.')
        exit()

def get_argument():

    #필요한 인수 = [생성할 데이터 수(각 달에서 생성할 데이터의 개수), 데이터를 저장할 저장공간]
    argument = []

    f_input = input('각 달에서 생성할 데이터의 수를 입력하시오(0 < N < 29)')

    if int(f_input) <= 0 or int(f_input) >= 29:
        print('잘못된 숫자입니다.')
        exit()
    else:
        argument.append(f_input)

    s_input = input('데이터를 저장할 경로를 입력하시오')

    argument.append(s_input)
    activate_generator(argument[0], argument[1])


def activate_generator(q_data, path):
    pass
    #구현하기: activation_model을 작동시키는 클래스 생성 코드 작성.

    #제네레이터 생성
    generator = Activate(int(q_data), path, file_paths)
    #모델 시작
    print('생성 시작했습니다.')
    generator.activate_model()



greeting()