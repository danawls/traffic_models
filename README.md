[Go To Engligh Ver](#english-ver)

# 🚦 GRU 모델과 교통류 이론을 결합한 교통 예측 연구

본 프로젝트는 대한민국의 교통 흐름 예측 정확도를 향상시키기 위해 **GRU(Gated Recurrent Unit)** 딥러닝 모델과 **전통적인 교통류 이론**을 결합한 **교통 예측 모델**을 개발한 연구입니다.

> 연구명: GRU모델과 교통류 이론을 결합한 교통 예측 모델의 개발에 관한 연구  
> 저자: 최우진 (Woojin Choi)  
> 연구 기간: 2024년 4월 - 2024년 8월

---

## 목차

- [연구 개요](#연구-개요)
- [연구 과정](#-연구-과정)
- [모델 구조](#-모델-구조)
- [실험 설정 및 성능 평가](#-실험-설정-및-성능-평가)
- [결론 및 기여](#-결론-및-기여)
- [디렉토리 구조](#-디렉토리-구조)
- [논문 파일](#-논문-파일)
- [저자 정보](#-저자-정보)

---

## 연구 개요

대한민국은 수도권의 인구 밀집과 급속한 자동차 보급으로 인해 심각한 교통 혼잡 문제에 직면하고 있습니다.  
기존 수리적 모델이나 단일 딥러닝 모델로는 **비선형적이고 동적인 교통 흐름을 충분히 설명하지 못한다는 한계**가 존재합니다

이에 본 연구는 **다양한 교통류 이론**과 **GRU 딥러닝 모델**을 결합하여, 예측 정확도를 높이고 실제 상황에 더 적합한 모델을 제시하고자 하였습니다.

---

## 연구 과정

### 연구 동기

기존 교통 예측 프로그램(지도앱 등)의 교통 예측 시스템이 부정확해 평소 곤경을 많이 겪음
교통 예측 프로그램, 모델을 개선하고자 하여 연구를 시작하였다.

### 연구 자료 탐색

연구 자료는 zotero 프로그램을 이용하여 탐색 및 정리하였다.

### 모델 개발

모델은 모두 파이썬을 이용하여 개발하였으며,
대부분 텐서플로우, scikit learn, 파이토치를 이용하였다.
데이터 처리에는 pandas와 dask를 이용하였다.

---

## 모델 구조

### 🔸 충격파 이론 + GRU

- 교통량 변화에 따라 발생하는 **충격파 강도**를 계산하여 입력 특성으로 사용
- ANN 또는 이동평균, 표준편차 기반 Regularization 적용

### 🔸 뉴웰의 관성 모델 + GRU

- 차량 간의 거리, 속도 반응을 **리셋/업데이트 게이트에 반영**
- **비선형 반응 함수**와 **시간 지연 파라미터**를 도입

### 🔸 다상 교통 이론 + GRU

- **상태 전이 행렬(S)**, **상태 영향 행렬(P)**, **상태 전이 영향 행렬(H)**을 통해
- 교통 흐름 상태 변화가 GRU의 은닉 상태에 반영되도록 설계

---

## 실험 설정 및 성능 평가

- **데이터 출처**: 국가교통정보센터 표준노드링크 데이터
- **분석 시기**: 2023년 1월, 4월, 7월, 10월 (기상 요소 반영)
- **모델 구성**: Hidden size 64, Epochs 1000, Optimizer: Adam, Loss: MSE

### 📊 평가 지표

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE
- MAPE
- R² Score

모든 결합 모델이 **기존 단일 GRU 및 통계적 모델보다 높은 예측 정확도**를 보였으며,  
특히 **중장기 예측에서는 충격파-GRU**, **단기 예측에서는 뉴웰-GRU**가 우수한 성능을 보였습니다.

---

## 📌 결론 및 기여

- 교통류 이론의 물리적 의미와 GRU 모델의 데이터 기반 학습을 통합
- 다양한 교통 상황(혼잡, 자유 흐름, 급격한 변화 등)에 대해 예측 정확도 향상
- 교통관리 시스템(ITS)에 실질적 기여 가능성 확인

---

## 디렉토리 구조

```bash
.
├── batch_program(자동데이터통합및처리)/     # 데이터를 불러오고, 통합 및 편집 한뒤 다시 저장해주는 자동 프로그램(파이썬)
├── cites/      # 레퍼런스들
├── data_combine/       # 일부 데이터들의 수동 데이터 통합(주피터 노트북)
├── dissertation/       # 논문 작성 폴더
├── experiences_notes/      # 연구 계획 및 과정 노트
├── get-daily-data/     # 일부 데이터 불러오기 및 확인
├── get-result/     # figure 생성 코드
├── missing_data/       #결측치 처리 코드
├── models/     # 모델 개발 코
├── table-figure/       # 표 및 사진
└── GRU_traffic.pdf     # 최종 논문

```

---

## 📄 논문 파일

- 📎 [논문 PDF 보기](/GRU_traffic.pdf)

---

## 👤 저자 정보

| 항목      | 내용                                                                                           |
| --------- | ---------------------------------------------------------------------------------------------- |
| 이름      | 최우진 (Woojin Choi)                                                                           |
| 소속      | NUTS (팀이메일: nuts.official.wj@gmail.com) (웹사이트: https://sanchojang123.wixsite.com/nuts) |
| 이메일    | sanchojang123@gmail.com                                                                        |
| 연구 기간 | 2024.04 ~ 2024.08                                                                              |

---

## English Ver

# 🚦 Research on Traffic Prediction by Integrating GRU Model and Traffic Flow Theory

This project aims to improve traffic flow prediction accuracy in South Korea by combining the **GRU (Gated Recurrent Unit)** deep learning model with **traditional traffic flow theories**.

> **Title**: A Study on the Development of a Traffic Prediction Model Combining GRU and Traffic Flow Theory  
> **Author**: Woojin Choi  
> **Research Period**: April 2024 – August 2024

---

## Table of Contents

- [Overview](#-overview)
- [Research Process](#-research-process)
- [Model Architecture](#-model-architecture)
- [Experiment Settings & Evaluation](#-experiment-settings--evaluation)
- [Conclusion & Contributions](#-conclusion--contributions)
- [Directory Structure](#-directory-structure)
- [Dissertation File](#-dissertation-file)
- [Author Info](#-author-info)

---

## 📌 Overview

South Korea is facing serious traffic congestion due to population concentration in the metropolitan area and a rapid increase in vehicles.  
Traditional mathematical or standalone deep learning models have limitations in explaining **nonlinear and dynamic traffic flow**.

This study combines **various traffic flow theories** with the **GRU deep learning model** to enhance prediction accuracy and develop a model that better reflects real-world situations.

---

## 🔬 Research Process

### Motivation

Frustration with the inaccuracy of current traffic prediction systems (e.g., navigation apps) inspired the research.  
The goal was to improve such models.

### Literature Review

Research materials were collected and managed using the **Zotero** reference manager.

### Model Development

All models were developed in **Python** using **TensorFlow**, **scikit-learn**, and **PyTorch**.  
Data processing was done using **pandas** and **dask**.

---

## 🧠 Model Architecture

### 🔸 Shockwave Theory + GRU

- Calculates **shockwave intensity** from traffic volume changes and uses it as input  
- Applies ANN, moving average, or standard deviation-based regularization

### 🔸 Newell’s Inertia Model + GRU

- Reflects **inter-vehicle distance and speed response** in the GRU's **reset/update gates**  
- Introduces **nonlinear response functions** and **time delay parameters**

### 🔸 Multi-phase Traffic Theory + GRU

- Designs GRU to reflect traffic flow transitions using:  
  - **State Transition Matrix (S)**  
  - **State Influence Matrix (P)**  
  - **State Transition Influence Matrix (H)**

---

## ⚙️ Experiment Settings & Evaluation

- **Data Source**: Korea Transport Data Standard Node-Link Dataset (KTDB)  
- **Analysis Periods**: January, April, July, and October 2023 (weather conditions included)  
- **Model Specs**: Hidden size 64, Epochs 1000, Optimizer: Adam, Loss: MSE

### 📊 Evaluation Metrics

- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE  
- MAPE  
- R² Score

All integrated models showed **higher prediction accuracy** than standalone GRU or statistical models.  
**Shockwave-GRU** performed best for **mid- to long-term predictions**, while **Newell-GRU** excelled in **short-term forecasts**.

---

## 🧾 Conclusion & Contributions

- Combined physical insights from traffic flow theory with GRU's data-driven learning  
- Improved prediction accuracy across various traffic conditions (e.g., congestion, free flow, sudden changes)  
- Demonstrated applicability to **Intelligent Transportation Systems (ITS)**

---

## 📁 Directory Structure

```bash
.
├── batch_program/             # Auto programs for loading, integrating, and saving traffic data (Python)
├── cites/                     # References and citations
├── data_combine/              # Manual data combination notebooks
├── dissertation/              # Dissertation drafts and files
├── experiences_notes/         # Research plans and notes
├── get-daily-data/            # Scripts for checking and loading data
├── get-result/                # Code for generating figures
├── missing_data/              # Missing value handling scripts
├── models/                    # Model development code
├── table-figure/              # Tables and images
└── GRU_traffic.pdf            # Final dissertation file
```

---

## 📄 Dissertation File

- 📎 [View PDF](/GRU_traffic.pdf)

---

## 👤 Author Info

| Field        | Details                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------ |
| Name         | Woojin Choi                                                                               |
| Affiliation  | NUTS (Team Email: nuts.official.wj@gmail.com) (Website: https://sanchojang123.wixsite.com/nuts) |
| Email        | sanchojang123@gmail.com                                                                   |
| Research Period | April 2024 – August 2024                                                                 |

---

