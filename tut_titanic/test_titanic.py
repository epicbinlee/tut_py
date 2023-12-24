import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------
def ticket_category(ticket):
    if 'A/' in ticket:
        return 'Category_A'
    elif 'PC' in ticket:
        return 'Category_PC'
    elif 'STON/O' in ticket:
        return 'Category_STON/O'
    else:
        return 'Category_Other'


label_encoder = LabelEncoder()
# ----------------------------------------------------------------
test_df = pd.read_csv(r'./datasets/test.csv')

# noinspection DuplicatedCode
print(test_df.shape)
print(test_df.columns)
print(test_df.isna().sum())

# ----------------------------------------------------------------

# 缺失值处理, Age缺失，Embarked缺失，Cabin缺失删除
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 特征工程
# Ticket
test_df['Ticket_Category'] = test_df['Ticket'].apply(ticket_category)
test_df.drop("Ticket", axis=1, inplace=True)
test_df['Ticket_Category'] = label_encoder.fit_transform(test_df['Ticket_Category'])
# Name
test_df['Title'] = test_df['Name'].apply(lambda x: re.search(r' ([A-Za-z]+)\.', x).group(1))
test_df['Title'] = label_encoder.fit_transform(test_df['Title'])
test_df['LastNameInitial'] = test_df['Name'].apply(lambda x: x.split(',')[0][0])
test_df['LastNameInitial'] = label_encoder.fit_transform(test_df['LastNameInitial'])
test_df['NameLength'] = test_df['Name'].apply(len)
test_df.drop(['Name'], axis=1, inplace=True)

test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])
test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])

ids = test_df['PassengerId'].values
test_df.drop("PassengerId", axis=1, inplace=True)

# ----------------------------------------------------------------
X = test_df

# 后续，您可以使用下面的代码来加载这个模型
model = lgb.Booster(model_file='lgb_model.txt')
y_pred = model.predict(X)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
print(y_pred_binary)
gender_df = pd.DataFrame({"PassengerId": ids, "Survived": y_pred_binary})
gender_df.to_csv(r'submission.csv', index=False)
