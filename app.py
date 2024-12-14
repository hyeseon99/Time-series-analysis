import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 11번까지 처리한 데이터를 가져옵니다
c = d.copy() 

# Label Encoding 적용할 범주형 변수들
label_encoders = {}
for column in ['city', 'state', 'type', 'IsWorkDay']:
    le = LabelEncoder()
    c[column] = le.fit_transform(c[column])
    label_encoders[column] = le

# 로그 정규화 적용 (로그 변환)
def log_transform(data):
    return np.log1p(data) 

def inverse_log_transform(data):
    return np.expm1(data)  

# 모든 가게에 대해 예측 결과 저장
results = []

# 각 가게에 대해 반복
for store_nbr in c['store_nbr'].unique():
    data_family = c[c['store_nbr'] == store_nbr]

    for family in data_family['family'].unique():
        family_data = data_family[data_family['family'] == family]

        #  훈련 데이터 설정
        train_data = family_data[(family_data['date'] >= '2013-01-01') & (family_data['date'] < '2017-08-15')]

        # 학습 데이터가 없는 경우 건너뛰기
        if train_data.empty:
            print(f"{store_nbr} 가게의 {family} 패밀리 학습 데이터가 없습니다.")
            continue

        # 모델 학습을 위한 데이터 준비
        X_train = train_data.drop(columns=['sales', 'date', 'family']).values
        y_train = train_data['sales'].values

        # 로그 정규화 적용 (특성 및 타겟)
        X_train_log = log_transform(X_train)
        y_train_log = log_transform(y_train)

        # XGBoost 데이터셋 생성
        dtrain = xgb.DMatrix(X_train_log, label=y_train_log)

        # XGBoost 모델 훈련
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',  # RMSE로 평가
            'eta': 0.1,  # 학습률
            'max_depth': 6,  # 최대 트리 깊이
            'min_child_weight': 3,  # 최소 가중치
            'subsample': 0.8,  # 데이터 샘플링 비율
            'colsample_bytree': 0.8,  # 피처 샘플링 비율
            'lambda': 1,  # L2 정규화
            'alpha': 0.1,  # L1 정규화
            'verbosity': 0
        }

        evals_result = {}

        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,  # 최대 부스팅 라운드
                evals=[(dtrain, 'train')],
                early_stopping_rounds=20,  # Early Stopping
                evals_result=evals_result,
                verbose_eval=False
            )
        except xgb.core.XGBoostError as e:
            print(f"모델 학습 중 오류 발생: {e}")
            continue

        # 예측할 데이터 준비 (8월 1일부터 15일까지)
        forecast_data = family_data[(family_data['date'] >= '2017-08-16') & (family_data['date'] <= '2017-08-31')]

        # 예측할 데이터가 없는 경우 건너뛰기
        if forecast_data.empty:
            print(f"{store_nbr} 가게의 {family} 패밀리 예측할 데이터가 없습니다.")
            continue

        # 예측을 위한 입력 데이터 ('sales', 'date', 'family' 제거)
        X_forecast = forecast_data.drop(columns=['sales', 'date', 'family']).values
        X_forecast_log = log_transform(X_forecast)  

        # XGBoost 모델을 사용한 판매량 예측
        dforecast = xgb.DMatrix(X_forecast_log)
        predicted_sales_log = model.predict(dforecast)

        # 예측 결과 역 로그 변환
        predicted_sales = inverse_log_transform(predicted_sales_log)

        # 음수를 0으로 변환
        predicted_sales = np.maximum(predicted_sales, 0)  

        # 결과 저장
        for sales, date in zip(predicted_sales, forecast_data['date']):
            results.append({
                'id': f"{store_nbr}_{family}_{date}",  
                'sales': sales,
            })

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results)

# 예측 결과를 CSV 파일로 저장
results_df.to_csv('c:\\data\\results6.csv', index=False)

# 모델 학습 결과 확인
print("모델 학습 완료 및 결과 저장 완료.")
