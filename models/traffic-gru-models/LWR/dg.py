import pandas as pd
import numpy as np

# 설정: 1달(30일) 동안의 5분 간격 데이터
minutes_per_day = 24 * 60
intervals = 5
days = 30
total_points = (minutes_per_day // intervals) * days

# 날씨 상태 정의
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Thunderstorm']


# 딥러닝 모델의 원리를 반영한 교통 데이터 생성 함수
def generate_realistic_traffic_data(timestamps):
    np.random.seed(42)  # 시드 고정으로 재현성 유지

    density = []
    speed = []
    weather = []
    is_holiday = []
    road_type = []
    event_impact = []

    for timestamp in timestamps:
        # 시간대와 요일에 따른 교통 특성 설정
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        is_weekend = day_of_week >= 5
        is_holiday_today = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% 확률로 공휴일
        current_weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.05, 0.05])  # 날씨 확률

        # 도로 유형 설정 (고속도로, 도심지, 일반도로)
        road = np.random.choice(['Highway', 'Urban', 'Suburban'], p=[0.3, 0.4, 0.3])
        road_type.append(road)

        # 기본 설정
        if road == 'Urban':
            base_density = np.random.normal(100, 20)  # 도심 밀도, 평균 100, 표준편차 20
            base_speed = np.random.normal(40, 10)  # 도심 속도, 평균 40, 표준편차 10
        elif road == 'Suburban':
            base_density = np.random.normal(50, 15)  # 외곽 밀도, 평균 50, 표준편차 15
            base_speed = np.random.normal(60, 15)  # 외곽 속도, 평균 60, 표준편차 15
        else:  # Highway
            base_density = np.random.normal(30, 10)  # 고속도로 밀도, 평균 30, 표준편차 10
            base_speed = np.random.normal(80, 20)  # 고속도로 속도, 평균 80, 표준편차 20

        # 시간대에 따른 패턴 (조건부 변화 반영)
        if 7 <= hour < 9:  # 출근 시간대
            base_density *= np.random.uniform(1.2, 1.5)
            base_speed *= np.random.uniform(0.6, 0.8)
        elif 16 <= hour < 19:  # 퇴근 시간대
            base_density *= np.random.uniform(1.3, 1.6)
            base_speed *= np.random.uniform(0.5, 0.7)
        elif 11 <= hour < 14:  # 점심 시간대, 교통량 증가 가능
            base_density *= np.random.uniform(1.1, 1.3)
            base_speed *= np.random.uniform(0.7, 0.9)
        else:  # 야간 시간대, 교통량 감소
            base_density *= np.random.uniform(0.8, 1.0)
            base_speed *= np.random.uniform(1.1, 1.3)

        # 주말 교통 패턴 조정 (조건부 변화 반영)
        if is_weekend:
            base_density *= np.random.uniform(0.6, 0.8)
            base_speed *= np.random.uniform(1.1, 1.3)

        # 날씨 영향 추가 (조건부 변화 반영)
        if current_weather == 'Rain':
            base_density *= np.random.uniform(1.1, 1.3)
            base_speed *= np.random.uniform(0.7, 0.9)
        elif current_weather == 'Snow':
            base_density *= np.random.uniform(1.3, 1.5)
            base_speed *= np.random.uniform(0.4, 0.6)
        elif current_weather == 'Fog':
            base_density *= np.random.uniform(1.0, 1.2)
            base_speed *= np.random.uniform(0.6, 0.8)
        elif current_weather == 'Thunderstorm':
            base_density *= np.random.uniform(1.4, 1.6)
            base_speed *= np.random.uniform(0.3, 0.5)

        # 공휴일 및 이벤트 영향 (조건부 변화 반영)
        if is_holiday_today:
            base_density *= np.random.uniform(0.4, 0.6)
            base_speed *= np.random.uniform(1.2, 1.5)

        # 임의의 교통 사고/이벤트 (1% 확률, 복잡한 상관관계 반영)
        event = np.random.rand() < 0.01
        if event:
            base_density *= np.random.uniform(1.5, 2.0)
            base_speed *= np.random.uniform(0.3, 0.5)
            event_impact.append(1)
        else:
            event_impact.append(0)

        # 임의의 변동 추가
        density.append(base_density + np.random.normal(0, 5))
        speed.append(base_speed + np.random.normal(0, 3))
        weather.append(current_weather)
        is_holiday.append(is_holiday_today)

    # 밀도 및 속도 제한
    density = np.clip(density, 0, 200)  # 밀도 제한
    speed = np.clip(speed, 0, 130)  # 속도 제한

    # 흐름 계산 (단순화된 모델)
    flow = np.array(density) * np.array(speed) / 100

    # 다음 시간 단계의 흐름을 예측 목표로 사용
    target = np.roll(flow, -1)
    target[-1] = np.nan  # 마지막 값은 NaN으로 설정 (예측 불가)

    return density, speed, flow, target, weather, is_holiday, road_type, event_impact



for i in range(4):
    # 타임스탬프 생성
    timestamps = pd.date_range(start='2024-01-01', periods=total_points, freq='5T')

    # 교통 데이터 생성
    density, speed, flow, target, weather, is_holiday, road_type, event_impact = generate_realistic_traffic_data(
        timestamps)

    # 데이터프레임 생성
    traffic_data = pd.DataFrame({
        'timestamp': timestamps,
        'density': density,
        'speed': speed,
        'flow': flow,
        'target': target,
        'weather': weather,
        'is_holiday': is_holiday,
        'road_type': road_type,
        'event_impact': event_impact
    })

    # CSV 파일로 저장
    traffic_data.to_csv(f'mnt/data/{i + 1}.csv', index=False)
    print(f"교통량 데이터가 '{i + 1}.csv'로 저장되었습니다.")