import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

import tools.trading_days as td

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
fund_data = fund_data[fund_data['日期'].apply(lambda x: isinstance(x, str) and '-' in x)]
fund_data['日期'] = pd.to_datetime(fund_data['日期'])
fund_data.set_index('日期', inplace=True)
print("基金净值数据加载完成，数据量：", fund_data.shape)

# 过滤9:30到14:00之间的数据
for name, df in index_dfs.items():
    index_dfs[name] = df.between_time('09:30', '14:00')

# 重命名列以包含指数名称
for name, df in index_dfs.items():
    df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != '日期'}, inplace=True)

# 将所有指数数据合并到一个DataFrame中
merged_df = pd.concat(index_dfs.values(), axis=1, join='inner')
print("合并后的数据框，数据量：", merged_df.shape)

# 为回归模型创建特征
features = ['收盘价(元)_中证银行', '收盘价(元)_万得银行业', '收盘价(元)_申万银行', '收盘价(元)_央企银行', '收盘价(元)_上证50']
merged_df['收盘价(元)_LOF基金_前一天'] = merged_df['收盘价(元)_LOF基金'].shift(1)
X = merged_df[features + ['收盘价(元)_LOF基金_前一天']].dropna()
y = merged_df['开盘价(元)_LOF基金'].loc[X.index]

# 将数据划分为训练集和测试集
split_date = '2024-04-09'
train_X = X.loc[X.index < split_date]
train_y = y.loc[X.index < split_date]
test_X = X.loc[X.index >= split_date]
test_y = y.loc[X.index >= split_date]

# 拟合回归模型
model = LinearRegression()
model.fit(train_X, train_y)

# 预测下一个交易日的基金开盘价
next_day_predictions = model.predict(test_X)

# 获取每个预测日期的 T+2 交易日
t_plus_2_dates = [td.find_t_plus_2(date) for date in test_X.index]

# 创建预测DataFrame
predictions_df = pd.DataFrame({
    'PredictionDate': t_plus_2_dates,  # 使用计算得到的 T+2 交易日
    'PredictedPrice': next_day_predictions
}, index=test_X.index)

# 显示预测结果
print(predictions_df.head())

# 转化 'PredictionDate' 为 datetime
predictions_df['PredictionDate'] = pd.to_datetime(predictions_df['PredictionDate'])

# 按天分组，找到每天最早的预测
earliest_indices = predictions_df.groupby(predictions_df['PredictionDate'].dt.date)['PredictionDate'].idxmin()

# 选择最早的预测
predictions_df_earliest = predictions_df.loc[earliest_indices]

# 将日期格式化为 'YYYY-MM-DD'
predictions_df_earliest['PredictionDate'] = predictions_df_earliest['PredictionDate'].dt.strftime('%Y-%m-%d')

# 重新索引基金净值数据，以匹配预测日期
fund_data_predictions = fund_data.reindex(predictions_df_earliest['PredictionDate'])

# 将预测结果与基金净值进行比较，制定交易策略
purchase_fee_rate = 0.015  # 假设申购费率为1.5%
redemption_fee_rate = 0.015  # 假设赎回费率为1.5%
signals = []
returns = []
cash = 100000  # 初始资金
shares = 0  # 初始份额

for i, row in predictions_df.iterrows():
    pred_price = row['PredictedPrice']
    prediction_date = row['PredictionDate'].date().strftime('%Y-%m-%d')  # Convert to 'YYYY-MM-DD' format
    actual_nav = fund_data_predictions.loc[prediction_date, '单位净值'] if prediction_date in fund_data_predictions.index else None

    if pd.notna(actual_nav):
        if pred_price > actual_nav * (1 + purchase_fee_rate):
            signal = '买入'
            # 执行买入操作，假设可以完全按预测价格买入
            if cash > 0:
                shares += cash / pred_price
                cash = 0
            print(f"{prediction_date}: 买入，预测价格：{pred_price}, 实际净值：{actual_nav}")
        elif pred_price < actual_nav - redemption_fee_rate:
            signal = '卖出'
            # 执行卖出操作，假设可以完全按预测价格卖出
            if shares > 0:
                cash += shares * pred_price
                shares = 0
            print(f"{prediction_date}: 卖出，预测价格：{pred_price}, 实际净值：{actual_nav}")
        else:
            signal = '持有'
    else:
        signal = '数据缺失'

    signals.append(signal)

# 将交易信号添加到DataFrame中
predictions_df['交易信号'] = signals

# 计算每日的资产总值
predictions_df['资产总值'] = cash + shares * predictions_df['PredictedPrice']

# 输出最终的资产总值
final_assets = predictions_df['资产总值'].iloc[-1]
print(f"初始资金: {100000}, 最终资产: {final_assets}, 收益率: {(final_assets - 100000) / 100000 * 100:.2f}%")

# 绘制资产总值变化图
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['PredictionDate'], predictions_df['资产总值'], label='资产总值')
plt.xlabel('日期')
plt.ylabel('资产总值 ($)')
plt.title('基金交易策略资产变化')
plt.legend()
plt.show()

# 计算每日收益率
predictions_df['收益率'] = predictions_df['资产总值'].pct_change()

# 绘制收益率曲线
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['PredictionDate'], predictions_df['收益率'], label='收益率')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.title('基金交易策略收益率变化')
plt.legend()
plt.show()

