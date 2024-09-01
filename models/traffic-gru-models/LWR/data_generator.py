import pandas as pd
import numpy as np

# 설정: 1달(30일) 동안의 5분 간격 데이터
minutes_per_day = 24 * 60
intervals = 5
days = 30
total_points = (minutes_per_day // intervals) * days

# 날씨 상태 정의
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog']


# 교통 데이터 생성 함수
def generate_realistic_traffic_data(timestamps):
    np.random.seed(42)  # 시드 고정으로 재현성 유지

    density = []
    speed = []
    weather = []
    is_holiday = []

    for timestamp in timestamps:
        # 시간대와 요일에 따른 교통 특성 설정
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        is_weekend = day_of_week >= 5
        is_holiday_today = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% 확률로 공휴일

        # 도심지와 외곽지역 구분 (도심지 비율 70%, 외곽 30%)
        is_urban = np.random.choice([True, False], p=[0.7, 0.3])

        # 기본 설정
        if is_urban:
            base_density = np.random.uniform(50, 150)  # 도심 밀도
            base_speed = np.random.uniform(20, 60)  # 도심 속도
        else:
            base_density = np.random.uniform(10, 70)  # 외곽 밀도
            base_speed = np.random.uniform(40, 100)  # 외곽 속도

        # 출근 시간대 (7-9 AM)
        if 7 <= hour < 10:
            base_density += np.random.uniform(30, 70) if is_urban else np.random.uniform(10, 30)
            base_speed -= np.random.uniform(10, 30) if is_urban else np.random.uniform(5, 15)
        # 낮 시간대 (10 AM - 3 PM)
        elif 10 <= hour < 16:
            base_density += np.random.uniform(10, 30) if is_urban else np.random.uniform(5, 15)
            base_speed -= np.random.uniform(5, 10) if is_urban else np.random.uniform(0, 5)
        # 퇴근 시간대 (4-7 PM)
        elif 16 <= hour < 20:
            base_density += np.random.uniform(40, 80) if is_urban else np.random.uniform(15, 40)
            base_speed -= np.random.uniform(20, 40) if is_urban else np.random.uniform(10, 20)
        # 야간 시간대 (8 PM - 6 AM)
        else:
            base_density += np.random.uniform(5, 10) if is_urban else np.random.uniform(0, 5)
            base_speed += np.random.uniform(10, 20) if is_urban else np.random.uniform(20, 40)

        # 주말 교통 감소
        if is_weekend:
            base_density *= 0.6
            base_speed *= 1.1

        # 날씨 영향 추가
        current_weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.1])
        if current_weather == 'Rain':
            base_density *= 1.2
            base_speed *= 0.8
        elif current_weather == 'Snow':
            base_density *= 1.3
            base_speed *= 0.6
        elif current_weather == 'Fog':
            base_density *= 1.1
            base_speed *= 0.7

        # 공휴일 및 이벤트
        if is_holiday_today:
            base_density *= 0.5
            base_speed *= 1.2

        # 임의의 교통 사고/이벤트 (1% 확률)
        if np.random.rand() < 0.01:
            base_density *= 1.5
            base_speed *= 0.5

        # 임의의 변동 추가
        density.append(base_density + np.random.uniform(-10, 10))
        speed.append(base_speed + np.random.uniform(-5, 5))
        weather.append(current_weather)
        is_holiday.append(is_holiday_today)

    density = np.clip(density, 0, 150)  # 밀도 제한
    speed = np.clip(speed, 0, 100)  # 속도 제한
    flow = np.array(density) * np.array(speed) / 100  # 흐름 계산 (단순화된 모델)

    target = np.roll(flow, -1)  # 다음 시간 단계의 흐름을 예측 목표로 사용
    target[-1] = np.nan  # 마지막 값은 NaN으로 설정 (예측 불가)

    return density, speed, flow, target, weather, is_holiday

for i in range(4):
    # 타임스탬프 생성
    timestamps = pd.date_range(start='2024-01-01', periods=total_points, freq='5T')

    # 교통 데이터 생성
    density, speed, flow, target, weather, is_holiday = generate_realistic_traffic_data(timestamps)

    # 데이터프레임 생성
    traffic_data = pd.DataFrame({
        'timestamp': timestamps,
        'density': density,
        'speed': speed,
        'flow': flow,
        'target': target,
        'weather': weather,
        'is_holiday': is_holiday
    })

    # CSV 파일로 저장
    traffic_data.to_csv(f'mnt/data/{i + 1}.csv', index=False)
    print(f"교통량 데이터가 '{i + 1}.csv'로 저장되었습니다.")