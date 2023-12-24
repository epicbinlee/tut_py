import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------
# 读取数据
train_df = pd.read_csv(r'./datasets/train.csv')

# ----------------------------------------------------------------
# 特征工程

# 取对数，消除长尾分布
train_df['area_sqm'] = np.log(train_df['area_sqm'])

# 计算房子年龄
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year
train_df['building_age'] = train_df['year'] - train_df['commence_date']

label_encoder = LabelEncoder()
train_df['storey_range_encoded'] = label_encoder.fit_transform(train_df['storey_range'])

cat_columns = ['type', 'flat_model', 'location']
train_df = pd.get_dummies(train_df, columns=cat_columns, drop_first=True, dtype=int)

train_label = train_df['price']
train_data = train_df.drop(['house_id', 'date', 'block', 'street', 'storey_range', 'price'], axis=1)

# ----------------------------------------------------------------
# 选择变量
X = train_data
y = train_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ----------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r_mse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {r_mse}')

joblib.dump(model, 'lr_model.pkl')
