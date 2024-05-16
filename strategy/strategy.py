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
    index_dfs[name]['日期'] = pd.to_datetime(index_dfs[name]['日期'])
    index_dfs[name].set_index('日期', inplace=True)
    print(f"{name} 数据加载完成，数据量：{index_dfs[name].shape}")

# 加载基金净值数据
fund_data_path = '../raw_data/鹏华中证银行A(160631.OF)-每日行情数据.xlsx'
fund_data = pd.read_excel(fund_data_path)

# 过滤掉非日期格式的行，并将日期列转换为datetime类型
fund_data = fund_data[fund_data['日期'].apply(lambda x: isinstance(x, str) and '-' in x)]
fund_data['日期'] = pd.to_datetime(fund_data['日期'])
fund_data.set_index('日期', inplace=True)
print("基金净值数据加载完成，数据量：", fund_data.shape)

# 检查基金净值数据的前几行
print("基金净值数据:", fund_data.head())

# 检查基金净值数据的索引范围
print("基金净值数据索引范围:", fund_data.index.min(), "到", fund_data.index.max())

# 过滤9:30到14:00之间的数据
for name, df in index_dfs.items():
    index_dfs[name] = df.between_time('09:30', '14:00')

# 重命名列以包含指数名称
for name, df in index_dfs.items():
    df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != '日期'}, inplace=True)

# 将所有指数数据合并到一个DataFrame中
merged_df = pd.concat(index_dfs.values(), axis=1, join='inner')
print("合并后的数据框，数据量：", merged_df.shape)

# 检查合并后的数据框的前几行
print("合并后的数据框:", merged_df.head())

# 检查索引格式
print("合并后数据框索引格式:", merged_df.index)

# 为回归模型创建特征
features = ['收盘价(元)_中证银行', '收盘价(元)_万得银行业', '收盘价(元)_申万银行', '收盘价(元)_央企银行',
            '收盘价(元)_上证50']
merged_df['收盘价(元)_LOF基金_前一天'] = merged_df['收盘价(元)_LOF基金'].shift(1)
X = merged_df[features + ['收盘价(元)_LOF基金_前一天']].dropna()
y = merged_df['开盘价(元)_LOF基金'].loc[X.index]

# 检查特征和目标变量的前几行
print("特征数据:", X.head())
print("目标数据:", y.head())

# 检查特征数据和目标数据的索引格式
print("特征数据索引格式:", X.index)
print("目标数据索引格式:", y.index)

# 将数据划分为训练集和测试集
split_date = '2024-04-09'
train_X = X.loc[X.index < split_date]
train_y = y.loc[y.index < split_date]
test_X = X.loc[X.index >= split_date]
test_y = y.loc[y.index >= split_date]

# 检查训练集和测试集的尺寸
print("训练集尺寸:", train_X.shape, train_y.shape)
print("测试集尺寸:", test_X.shape, test_y.shape)

# 检查训练集和测试集的索引范围
print("训练集索引范围:", train_X.index.min(), "到", train_X.index.max())
print("测试集索引范围:", test_X.index.min(), "到", test_X.index.max())

# 拟合回归模型
model = LinearRegression()
model.fit(train_X, train_y)

# 预测LOF基金开盘价
test_predictions = model.predict(test_X)

# 检查测试集索引
print("测试集索引:", test_X.index)

# 将测试集索引转换为日期格式
test_X_dates = test_X.index.normalize()

# 获取每天的第一个数据点
first_test_X = test_X.groupby(test_X_dates).first()
first_predictions = pd.Series(test_predictions, index=test_X_dates).groupby(level=0).first()

# 确保测试集索引存在于基金净值数据中
fund_data_test = fund_data.loc[fund_data.index.intersection(first_test_X.index)]
print("对齐后的基金净值数据:", fund_data_test.head())

# 确保 fund_data 和 first_predictions 长度匹配
if len(fund_data_test) != len(first_predictions):
    min_len = min(len(fund_data_test), len(first_predictions))
    fund_data_test = fund_data_test.iloc[:min_len]
    first_predictions = first_predictions.iloc[:min_len]

# 使用基金净值
fund_net_value = fund_data_test['单位净值']
predicted_open_price = pd.Series(first_predictions, index=fund_data_test.index)

# 检查对齐后的数据
print("对齐后的基金净值数据:", fund_net_value.head())
print("对齐后的预测开盘价:", predicted_open_price.head())

# 考虑交易费用的决策制定
purchase_fee_rate = 0.012  # 假设申购费率为1.2%
redemption_fee_rate = 0.015  # 假设赎回费率为1.5%

signals = []
daily_returns = []

# 初始资金
initial_cash = 1000000
cash = initial_cash
position = 0

for i in range(len(predicted_open_price)):
    purchase_fee = fund_net_value.iloc[i] * purchase_fee_rate
    redemption_fee = fund_net_value.iloc[i] * redemption_fee_rate
    if predicted_open_price.iloc[i] > fund_net_value.iloc[i] + purchase_fee:
        signals.append('在一级市场申购，二级市场卖出')
        # 卖出操作
        if position > 0:
            cash += position * predicted_open_price.iloc[i]
            position = 0
    elif predicted_open_price.iloc[i] < fund_net_value.iloc[i] - redemption_fee:
        signals.append('在一级市场赎回，二级市场买入')
        # 买入操作
        if cash > predicted_open_price.iloc[i]:
            position = cash / predicted_open_price.iloc[i]
            cash = 0
    else:
        signals.append('保持观望')

    # 计算每日收益
    daily_return = cash + position * predicted_open_price.iloc[i] - initial_cash
    daily_returns.append(daily_return)

# 添加信号到DataFrame
decision_df = pd.DataFrame(index=fund_data_test.index)
decision_df['预测开盘价'] = predicted_open_price
decision_df['基金净值'] = fund_net_value
decision_df['交易信号'] = signals
decision_df['每日收益'] = daily_returns

# 显示交易信号
print(decision_df[['预测开盘价', '基金净值', '交易信号', '每日收益']].head())
decision_df[['预测开盘价', '基金净值', '交易信号', '每日收益']].to_excel("./output/交易收益.xlsx")

# 检查收益数据
print("每日收益数据:", decision_df['每日收益'].head())

# 绘制累计收益曲线
plt.figure(figsize=(14, 7))
plt.plot(decision_df.index, decision_df['每日收益'].cumsum(), label='累计收益')
plt.xlabel('日期')
plt.ylabel('累计收益')
plt.title('基于预测的交易策略累计收益')
plt.legend()
plt.show()
