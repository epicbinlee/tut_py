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
train_df = pd.read_csv(r'./datasets/train.csv')
# test_df = pd.read_csv(r'./datasets/test.csv')
# gender_df = pd.read_csv(r'./datasets/gender_submission.csv')

# noinspection DuplicatedCode
# 查看基础内容
print(train_df.shape)
print(train_df.columns)
print(train_df.isna().sum())

# ----------------------------------------------------------------

# 缺失值处理, Age缺失，Embarked缺失，Cabin缺失删除
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df.drop("Cabin", axis=1, inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# 特征工程
# Ticket
train_df['Ticket_Category'] = train_df['Ticket'].apply(ticket_category)
train_df.drop("Ticket", axis=1, inplace=True)
train_df['Ticket_Category'] = label_encoder.fit_transform(train_df['Ticket_Category'])
# Name
train_df['Title'] = train_df['Name'].apply(lambda x: re.search(r' ([A-Za-z]+)\.', x).group(1))
train_df['Title'] = label_encoder.fit_transform(train_df['Title'])
train_df['LastNameInitial'] = train_df['Name'].apply(lambda x: x.split(',')[0][0])
train_df['LastNameInitial'] = label_encoder.fit_transform(train_df['LastNameInitial'])
train_df['NameLength'] = train_df['Name'].apply(len)
train_df.drop(['Name'], axis=1, inplace=True)

train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# label
label = train_df['Survived'].values
train_df.drop("Survived", axis=1, inplace=True)
train_df.drop("PassengerId", axis=1, inplace=True)

# ----------------------------------------------------------------
# 数据集构造
X = train_df
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',  # 目标函数，指定为二分类问题
    'metric': 'binary_logloss',  # 模型性能评估指标，通常选择对数损失（logloss）用于二分类
    'boosting_type': 'gbdt',  # 提升树的类型，通常使用梯度提升决策树（Gradient Boosting Decision Tree）
    'num_leaves': 8,  # 每棵树的叶子节点数，控制树的复杂度，可以根据需要调整
    'learning_rate': 0.02,  # 学习率，控制每一步的步长，通常需要调整此参数
    'feature_fraction': 0.9,  # 每棵树分裂时使用的特征的比例，用于控制过拟合，可以根据需要调整
    'verbose': 1  # 控制模型训练时的输出信息级别，1 表示显示进度信息
    # 其他参数可以根据具体需求添加和调整
}


# 创建一个回调函数示例
def custom_callback(env):
    # 在每一轮训练结束后打印当前的训练损失
    print("Round {}, Train's binary_logloss: {}".format(env.iteration, env.evaluation_result_list))


num_round = 1000
early_stopping_rounds = 10

model = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=[custom_callback, lgb.early_stopping(stopping_rounds=10)])

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

# 保存模型到文件
model.save_model('lgb_model.txt')
