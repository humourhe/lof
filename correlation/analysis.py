import pandas as pd
import math
import matplotlib.pyplot as plt
import tools.trading_days as td
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
features = ['收盘价(元)_中证银行', '收盘价(元)_万得银行业', '收盘价(元)_申万银行', '收盘价(元)_上证50',
            '收盘价(元)_LOF基金']

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

# 定义要评估的模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'SVR': SVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'GBDT': GradientBoostingRegressor()
}

# 评估每个模型
results = {}
for name, model in models.items():
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)

    train_mse = mean_squared_error(train_y, train_predictions)
    test_mse = mean_squared_error(test_y, test_predictions)
    train_r2 = r2_score(train_y, train_predictions)
    test_r2 = r2_score(test_y, test_predictions)

    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }

# 打印指标
for name, result in results.items():
    print(f"{name} 训练集MSE: {result['train_mse']}, 测试集MSE: {result['test_mse']}")
    print(f"{name} 训练集R²: {result['train_r2']}, 测试集R²: {result['test_r2']}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(14, 7))
plt.plot(test_y.index, test_y, label='实际LOF基金净值')

for name, result in results.items():
    plt.plot(test_y.index, result['test_predictions'], label=f'{name} 预测值', linestyle='dashed')

plt.xlabel('日期')
plt.ylabel('LOF基金净值')
plt.title('实际LOF基金净值与预测值的基金净值对比')
plt.legend()
plt.show()
