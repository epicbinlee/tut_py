import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------
# 读取数据
test_df = pd.read_csv(r'./datasets/test.csv')

# ----------------------------------------------------------------
# 特征工程

# 取对数，消除长尾分布
# noinspection DuplicatedCode
test_df['area_sqm'] = np.log(test_df['area_sqm'])

# 计算房子年龄
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['building_age'] = test_df['year'] - test_df['commence_date']

label_encoder = LabelEncoder()
test_df['storey_range_encoded'] = label_encoder.fit_transform(test_df['storey_range'])

cat_columns = ['type', 'flat_model', 'location']
test_df = pd.get_dummies(test_df, columns=cat_columns, drop_first=True, dtype=int)

housing_id = test_df['house_id']
test_data = test_df.drop(['house_id', 'date', 'block', 'street', 'storey_range'], axis=1)

# ----------------------------------------------------------------

loaded_model = joblib.load('lr_model.pkl')
predictions = loaded_model.predict(test_data)

result = pd.DataFrame({'house_id': housing_id, 'price': predictions})
result.to_csv('result.csv', index=False)

print('predictions:', predictions)
