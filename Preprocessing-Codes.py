#데이터 불러오기
import pandas as pd
train = pd.read_csv("c:\\data\\train.csv")
test = pd.read_csv("c:\\data\\test.csv")
stores = pd.read_csv("c:\\data\\stores.csv")
transactions = pd.read_csv("c:\\data\\transactions.csv").sort_values(["store_nbr", "date"])
oil = pd.read_csv("c:\\data\\oil_updated.csv")

train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)

train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

---
#개업전? 가게 전꺼 데이터 날리기 방법1 
print(train.shape)
train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]
train.shape 

---

c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family","store_nbr"])
c = c[c.sales == 0]
c
---

#판매량이 쭉 0인것들 제외
import gc 
print(train.shape)
# Anti Join
outer_join = train.merge(c[c.sales == 0].drop("sales",axis = 1), how = 'outer', indicator = True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
del outer_join
gc.collect()
train.shape

---

#테스트 데이터 대비 0채워넣기
zero_prediction = []
for i in range(0,len(c)):
    zero_prediction.append(
        pd.DataFrame({
            "date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
            "store_nbr":c.store_nbr.iloc[i],
            "family":c.family.iloc[i],
            "sales":0
        })
    )
zero_prediction = pd.concat(zero_prediction)
del c
gc.collect()
zero_prediction
---

# one_hot_encoder 함수 정의
import numpy as np
pd.set_option('display.max_columns',None)
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

# Import holidays data
holidays = pd.read_csv("c:\\data\\holidays_events.csv")
holidays["date"] = pd.to_datetime(holidays.date)

# Process transferred holidays
tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1).reset_index(drop=True)
tr = pd.concat([tr1, tr2], axis=1)
tr = tr.iloc[:, [5, 1, 2, 3, 4]]

holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
holidays = pd.concat([holidays, tr]).reset_index(drop=True)

# Additional holidays and bridge holidays
holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', '')
holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])
holidays["description"] = holidays["description"].str.replace("Puente ", "")
holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

# Work Day Holidays
work_day = holidays[holidays.type == "Work Day"]
holidays = holidays[holidays.type != "Work Day"]

# Split into events, national, regional, local holidays
events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis=1).rename({"description": "events"}, axis=1)
holidays = holidays[holidays.type != "Event"].drop("type", axis=1)

regional = holidays[holidays.locale == "Regional"].rename({"locale_name": "state", "description": "holiday_regional"}, axis=1).drop("locale", axis=1).drop_duplicates()
national = holidays[holidays.locale == "National"].rename({"description": "holiday_national"}, axis=1).drop(["locale", "locale_name"], axis=1).drop_duplicates()
local = holidays[holidays.locale == "Local"].rename({"description": "holiday_local", "locale_name": "city"}, axis=1).drop("locale", axis=1).drop_duplicates()

# Merge with train and test
d = pd.merge(pd.concat([train, test]), stores)
d["store_nbr"] = d["store_nbr"].astype("int8")

# Merge holidays data
d = pd.merge(d, national, how="left")
d = pd.merge(d, regional, how="left", on=["date", "state"])
d = pd.merge(d, local, how="left", on=["date", "city"])
d = pd.merge(d, work_day[["date", "type"]].rename({"type": "IsWorkDay"}, axis=1), how="left")

# Handle events data
events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)
events, events_cat = one_hot_encoder(events, nan_as_category=False)
events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
events = events.drop(239)

d = pd.merge(d, events, how="left")
d[events_cat] = d[events_cat].fillna(0)

# New features
d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)
d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha', 'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

# One-hot encoding for holidays
holidays, holidays_cat = one_hot_encoder(d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category=False)
d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis=1), holidays], axis=1)

# Final cleanup
he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + d.columns[d.columns.str.startswith("local")].tolist()
d[he_cols] = d[he_cols].astype("int8")
d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols
gc.collect()

d.head(10)

---
def create_date_features(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = df.date.dt.isocalendar().week.astype("int8")  # 수정된 부분
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df["season"] = np.where(df.month.isin([6, 7, 8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df

# 날짜 컬럼 생성
d = create_date_features(d)

# Workday column 수정
d["workday"] = np.where((d.holiday_national_binary == 1) | 
                        (d.holiday_local_binary == 1) | 
                        (d.holiday_regional_binary == 1) | 
                        (d['day_of_week'].isin([6, 7])), 0, 1).astype("int8")

# Wages in the public sector are paid every two weeks on the 15th and on the last day of the month. 
d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

print(d.head(19))
print(len(d))
---
import pandas as pd

# Simple Moving Average(SMA) 계산 함수
def sma_features(dataframe, windows, lags):
    dataframe = dataframe.copy()
    for window in windows:
        for lag in lags:
            dataframe[f'SMA{window}_sales_lag{lag}'] = dataframe.groupby(["store_nbr", "family"])['sales'] \
                .transform(lambda x: x.shift(lag).rolling(window=window).mean())
    return dataframe

# 사용 예시
windows = [20, 30, 45, 60, 90, 120, 365, 730, 241, 23, 457, 169]  # 다양한 주기의 SMA
lags = [16, 30, 60, 90, 241, 23, 457, 169]  # 다양한 지연 변수
d = sma_features(d, windows, lags)

# 결과 확인
print(d.head())  # 상위 5개 행 출력

---
# EWM 생성하는 함수
def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

# 사용 예시
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [16, 30, 60, 90, 241, 23, 457, 169]  # 다양한 주기를 추가

d = ewm_features(d, alphas, lags)

# 결과 확인
print(d.head())  # 상위 5개 행 출력


---
def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store_nbr", "family"])['sales'].shift(lag)
    return dataframe

# 사용 예시
lags = [16, 30, 60, 90, 241, 23, 457, 169]  # 1일, 7일, 14일, 28일 지연 변수 추가
d = lag_features(d, lags)
print(d.head())  # 상위 5개 행 출력
---









