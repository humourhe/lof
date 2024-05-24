import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tools.trading_days as td
import matplotlib.font_manager as fm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 查找并设置中文字体
zh_font_path = 'C:/Windows/Fonts/simhei.ttf'  # 根据实际情况设置路径

# 更新字体配置
prop = fm.FontProperties(fname=zh_font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 确保负号可以正确显示

# 加载高频交易数据
index_paths = {
    '上证50': '../raw_data/000016.SH-上证50指数.xlsx',
    '沪深300': '../raw_data/000300.SH-沪深300指数.xlsx',
    '金融地产': '../raw_data/399975.SZ-中证金融地产指数.xlsx',
    '中证银行': '../raw_data/399986.SZ-中证银行指数.xlsx',
    '申万银行': '../raw_data/801780.SI-银行(申万)指数.xlsx',
    '万得银行业': '../raw_data/886052.WI-万得银行业指数.xlsx',
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

# 将数据按照日期重新采样，然后获取每天的最后一条记录
merged_df = merged_df.resample('D').last()

# 删除所有的NaN值
merged_df = merged_df.dropna(how='all')

print("合并后的数据框，数据量：", merged_df.shape)
print("合并后的数据框，数据时间范围：", merged_df.index.min(), "to", merged_df.index.max())

# 为回归模型创建特征
features = ['收盘价(元)_中证银行', '收盘价(元)_万得银行业', '收盘价(元)_申万银行', '收盘价(元)_上证50', '收盘价(元)_LOF基金']

# 创建目标变量为T+2日期的基金净值
merged_df['T+2日期'] = merged_df.index.map(lambda x: td.find_t_plus_2(x).strftime('%Y-%m-%d'))
fund_data['日期'] = fund_data.index.strftime('%Y-%m-%d')

# 使用日期对齐进行合并
merged_df = merged_df.join(fund_data.set_index('日期')['单位净值'], on='T+2日期', rsuffix='_T+2')
merged_df.dropna(subset=['单位净值'], inplace=True)  # 去除无效的目标变量

X = merged_df[features]
y = merged_df['单位净值']

# 将数据划分为80%训练集和20%测试集
split_date = '2024-04-02'
train_X = X.loc[X.index < split_date]
train_y = y.loc[y.index < split_date]
test_X = X.loc[X.index >= split_date]
test_y = y.loc[y.index >= split_date]

# 标准化特征值
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# 拟合Ridge回归模型
model = Ridge(alpha=1.0)  # 选择合适的alpha值
model.fit(train_X_scaled, train_y)

# 预测LOF基金净值
train_predictions = model.predict(train_X_scaled)
test_predictions = model.predict(test_X_scaled)

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
plt.plot(test_y.index, test_y, label='实际LOF基金净值')
plt.plot(test_y.index, test_predictions, label='预测LOF基金净值', linestyle='dashed')
plt.xlabel('日期')
plt.ylabel('LOF基金净值')
plt.title('实际LOF基金净值与预测值的基金净值对比')
plt.legend()
plt.show()

# 预测T+2日期的基金净值
next_day_predictions = model.predict(test_X_scaled)

# 创建预测DataFrame，保留原始索引
predictions_df = pd.DataFrame({
    'PredictionDate': test_X.index.map(lambda x: td.find_t_plus_2(x)),  # 使用计算得到的 T+2 交易日
    'PredictedPrice': next_day_predictions
}, index=test_X.index)

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
cash = 100000  # 初始100000元本金
shares = 0  # 初始份额

# 添加每日资产记录
daily_stocks = []
daily_shares = []
daily_assets = []

# 初始股票数量
stocks = cash / merged_df.loc[predictions_df.index.min()]["收盘价(元)_LOF基金"]

for i, row in predictions_df.iterrows():
    pred_price = row['PredictedPrice']
    prediction_date = row['PredictionDate'].date().strftime('%Y-%m-%d')  # 将日期转换为'YYYY-MM-DD'格式
    actual_nav = fund_data_predictions.loc[prediction_date, '单位净值'] if prediction_date in fund_data_predictions.index else None

    if pd.notna(actual_nav):  # 检查实际净值是否存在
        if merged_df.loc[i]["收盘价(元)_LOF基金"] > pred_price * (1 + purchase_fee_rate):  # 如果收盘价高于考虑申购费后的预测净值
            signal = '卖出股票，买入LOF基金'
            if stocks > 0:  # 如果有股票可用，执行卖出股票，买入LOF基金操作
                shares += stocks / actual_nav  # 计算实际可以买入的基金份额
                stocks = 0  # 所有股票用于买入
            print(f"{prediction_date}: 卖出股票，买入LOF基金，预测基金净值：{pred_price}, 实际基金净值：{actual_nav}, 股票价格：{merged_df.loc[i]["收盘价(元)_LOF基金"]}")
        elif merged_df.loc[i]["收盘价(元)_LOF基金"] < pred_price * (1 - redemption_fee_rate):  # 如果收盘价格低于考虑赎回费后的预测净值
            signal = '买入股票，卖出LOF基金'
            if shares > 0:  # 如果有份额可用，执行卖出操作
                stocks += (shares * actual_nav * (1 - redemption_fee_rate)) / merged_df.loc[i]["收盘价(元)_LOF基金"]  # 计算卖出份额获得的股票
                shares = 0  # 卖出所有份额
            print(f"{prediction_date}: 买入股票，卖出LOF基金，预测基金净值：{pred_price}, 实际基金净值：{actual_nav}, 股票价格：{merged_df.loc[i]["收盘价(元)_LOF基金"]}")
        else:
            signal = '持有'  # 如果预测价格与实际净值相差不大，则持有不动
    else:
        signal = '数据缺失'  # 如果实际净值不存在，标记为数据缺失

    signals.append(signal)  # 将交易信号添加到信号列表中

    # 记录每日现金、持股和资产总值
    daily_stocks.append(stocks)  # 记录每日现金余额
    daily_shares.append(shares)  # 记录每日持有份额
    if not math.isnan(actual_nav):
        daily_asset = shares * actual_nav + stocks * merged_df.loc[i]["收盘价(元)_LOF基金"] # 计算总资产价值
    else:
        daily_asset = stocks * merged_df.loc[i]["收盘价(元)_LOF基金"] # 如果实际净值缺失，则将总资产视为现金

    daily_assets.append(daily_asset)  # 记录每日总资产价值


# 将交易信号添加到DataFrame中
predictions_df['交易信号'] = signals

# 计算每日的资产总值
predictions_df['资产总值'] = daily_assets

# 输出最终的资产总值
final_assets = predictions_df['资产总值'].iloc[-1]
print(f"初始资金: {cash}, 最终资产: {final_assets}, 收益率: {(final_assets - cash) / cash * 100:.2f}%")

# 绘制资产总值变化图
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['PredictionDate'], predictions_df['资产总值'], label='资产总值')
plt.xlabel('日期')
plt.ylabel('资产总值 ($)')
plt.title('基金交易策略资产变化')
plt.legend()
plt.show()

# 计算每日的累计收益
predictions_df['累计收益'] = (predictions_df['资产总值'] / predictions_df['资产总值'].iloc[0]) - 1

# 绘制累计收益曲线
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['PredictionDate'], predictions_df['累计收益'], label='累计收益')
plt.xlabel('日期')
plt.ylabel('累计收益')
plt.title('基金交易策略累计收益变化')
plt.legend()
plt.show()

# 计算每日盈亏
predictions_df['每日盈亏'] = predictions_df['资产总值'].diff().fillna(0)

# 绘制每日盈亏曲线
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['PredictionDate'], predictions_df['每日盈亏'], label='每日盈亏')
plt.xlabel('日期')
plt.ylabel('每日盈亏 ($)')
plt.title('基金交易策略每日盈亏变化')
plt.legend()
plt.show()

# 回测结果分析
# 计算年化收益率、夏普比率、最大回撤和胜率
annual_return = (predictions_df['资产总值'].iloc[-1] / predictions_df['资产总值'].iloc[0]) ** (252.0 / len(predictions_df)) - 1
daily_returns = predictions_df['资产总值'].pct_change().fillna(0)
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

# 计算最大回撤
roll_max = predictions_df['资产总值'].cummax()
daily_drawdown = predictions_df['资产总值'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

# 计算胜率
num_trades = len(predictions_df[predictions_df['交易信号'] != '持有'])
num_wins = len(predictions_df[(predictions_df['交易信号'] != '持有') & (predictions_df['每日盈亏'] > 0)])
win_rate = num_wins / num_trades if num_trades > 0 else 0

# 打印回测结果
print(f"年化收益率: {annual_return * 100:.2f}%")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown * 100:.2f}%")
print(f"胜率: {win_rate * 100:.2f}%")
