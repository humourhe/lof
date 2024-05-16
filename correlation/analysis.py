import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# 将数据加载到pandas DataFrames
index_dfs = {}
for name, path in index_paths.items():
    index_dfs[name] = pd.read_excel(path)

# 将日期列转换为datetime
for name, df in index_dfs.items():
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

# 过滤出9:30至14:00之间的数据
for name, df in index_dfs.items():
    index_dfs[name] = df.between_time('09:30', '14:00')

# 重命名列以包含指数名称
for name, df in index_dfs.items():
    df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != '日期'}, inplace=True)

# 将所有指数数据合并到一个DataFrame
merged_df = pd.concat(index_dfs.values(), axis=1, join='inner')

# 仅提取收盘价进行相关性分析
closing_prices = merged_df[[col for col in merged_df.columns if '收盘价' in col]]

# 计算相关性矩阵
correlation_matrix = closing_prices.corr()

# 提取相关的相关性
relevant_correlations = correlation_matrix['收盘价(元)_LOF基金'].sort_values(ascending=False)

# 绘制相关性矩阵
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('LOF基金和指数收盘价的相关性矩阵')
plt.show()

# 显示与LOF基金收盘价相关的相关性
print("与LOF基金收盘价相关的相关性:")
print(relevant_correlations)