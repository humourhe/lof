import pandas as pd
import matplotlib.pyplot as plt
import tools.trading_days as td
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
    index_dfs[name]['日期'] = pd.to_datetime(index_dfs[name]['日期'])  # 将日期列转换为datetime类型
    index_dfs[name].set_index('日期', inplace=True)  # 将日期列设为索引
    print(f"{name} 数据加载完成，数据量：{index_dfs[name].shape}")

# 加载基金净值数据
fund_data_path = '../raw_data/鹏华中证银行A(160631.OF)-每日行情数据.xlsx'
fund_data = pd.read_excel(fund_data_path)
fund_data = fund_data[fund_data['日期'].apply(lambda x: isinstance(x, str) and '-' in x)]
fund_data['日期'] = pd.to_datetime(fund_data['日期'])  # 将日期列转换为datetime类型
fund_data.set_index('日期', inplace=True)  # 将日期列设为索引
print("基金净值数据加载完成，数据量：", fund_data.shape)

# 过滤9:30到14:00之间的数据
for name, df in index_dfs.items():
    index_dfs[name] = df.between_time('09:30', '14:00')  # 过滤指定时间段的数据

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

# 标准化特征值
scaler = StandardScaler()
# 将数据划分为80%训练集和20%测试集
split_date = '2024-04-02'
train_X = scaler.fit_transform(X.loc[X.index < split_date])
test_X = scaler.transform(X.loc[X.index >= split_date])
train_y = y.loc[y.index < split_date]
test_y = y.loc[y.index >= split_date]


# 使用交叉验证来选择最佳参数
def tune_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_X, train_y)
    best_model = grid_search.best_estimator_
    return best_model


# 定义要评估的模型及其参数网格
models_param_grid = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge Regression': (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}),
    'Lasso Regression': (Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1, 10]}),  # 增加最大迭代次数
    'SVR': (SVR(), {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1]}),
    'Decision Tree': (DecisionTreeRegressor(), {'max_depth': [3, 5, 7, 9]}),
    'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
    'GBDT': (GradientBoostingRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]})
}

# 评估每个模型
results = {}
for name, (model, param_grid) in models_param_grid.items():
    if param_grid:
        model = tune_model(model, param_grid)  # 如果有参数网格，则进行调优
    else:
        model.fit(train_X, train_y)  # 对于线性回归，无需调优，直接训练模型

    # 进行预测
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)

    # 计算指标
    train_mse = mean_squared_error(train_y, train_predictions)
    test_mse = mean_squared_error(test_y, test_predictions)
    train_r2 = r2_score(train_y, train_predictions)
    test_r2 = r2_score(test_y, test_predictions)

    # 存储结果
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
