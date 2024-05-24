import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载高频交易数据
index_paths = {
    '上证50': '../raw_data/000016.SH-上证50指数.xlsx',
    '沪深300': '../raw_data/000300.SH-沪深300指数.xlsx',
    '金融地产': '../raw_data/399975.SZ-中证金融地产指数.xlsx',
    '中证银行': '../raw_data/399986.SZ-中证银行指数.xlsx',
    '申万银行': '../raw_data/801780.SI-银行(申万)指数.xlsx',
    '万得银行业': '../raw_data/886052.WI-万得银行业指数.xlsx',
    '央企银行': '../raw_data/8841787.WI-银行央企指数.xlsx',
    'LOF基金': '../raw_data/160631.SZ-银行LOF基金.xlsx'
}

# 将数据加载到pandas DataFrame中
index_dfs = {}
for name, path in index_paths.items():
    index_dfs[name] = pd.read_excel(path)

# 将日期列转换为datetime类型
for name, df in index_dfs.items():
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

# 过滤9:30到14:00之间的数据
for name, df in index_dfs.items():
    index_dfs[name] = df.between_time('09:30', '14:00')

# 重命名列以包含指数名称
for name, df in index_dfs.items():
    df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != '日期'}, inplace=True)

# 将所有指数数据合并到一个DataFrame中
merged_df = pd.concat(index_dfs.values(), axis=1, join='inner')

# 为回归模型创建特征
features = ['收盘价(元)_中证银行', '收盘价(元)_万得银行业', '收盘价(元)_申万银行', '收盘价(元)_央企银行', '收盘价(元)_上证50']
merged_df['收盘价(元)_LOF基金_前一天'] = merged_df['收盘价(元)_LOF基金'].shift(1)
X = merged_df[features + ['收盘价(元)_LOF基金_前一天']].dropna()
y = merged_df['开盘价(元)_LOF基金'].loc[X.index]

# 将数据划分为训练集和测试集,80%数据集和20%测试集
split_date = '2024-04-09'
train_X = X.loc[X.index < split_date]
train_y = y.loc[y.index < split_date]
test_X = X.loc[X.index >= split_date]
test_y = y.loc[y.index >= split_date]

# 线性回归模型拟合回归模型
model = LinearRegression()
model.fit(train_X, train_y)

# 预测LOF基金开盘价
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

# 计算指标
train_mse = mean_squared_error(train_y, train_predictions)
test_mse = mean_squared_error(test_y, test_predictions)
train_r2 = r2_score(train_y, train_predictions)
test_r2 = r2_score(test_y, test_predictions)

# 打印指标
print(f"训练集MSE: {train_mse}")
print(f"测试集MSE: {test_mse}")
print(f"训练集R^2: {train_r2}")
print(f"测试集R^2: {test_r2}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(14, 7))
plt.plot(test_y.index, test_y, label='实际LOF基金开盘价')
plt.plot(test_y.index, test_predictions, label='预测LOF基金开盘价', linestyle='dashed')
plt.xlabel('日期')
plt.ylabel('LOF基金开盘价')
plt.title('实际值与预测值的LOF基金开盘价对比')
plt.legend()
plt.show()
