### 结果分析

根据各模型在训练集和测试集上的MSE和R²指标，以及绘制的拟合曲线图，我们可以分析每个模型的表现：

#### Linear Regression
- **训练集MSE**: 0.0001107
- **测试集MSE**: 0.0006246
- **训练集R²**: 0.9402
- **测试集R²**: -1.5946
- **分析**: 训练集上表现较好，但测试集上表现差，说明模型过拟合。

#### Ridge Regression
- **训练集MSE**: 0.0001160
- **测试集MSE**: 0.0000947
- **训练集R²**: 0.9373
- **测试集R²**: 0.6065
- **分析**: 在测试集上的表现较好，R²为正，是所有模型中表现最好的，说明通过正则化有效地减少了过拟合。

#### Lasso Regression
- **训练集MSE**: 0.0002311
- **测试集MSE**: 0.0004608
- **训练集R²**: 0.8752
- **测试集R²**: -0.9142
- **分析**: 训练集表现一般，测试集表现较差，模型可能仍然存在一定的过拟合。

#### SVR
- **训练集MSE**: 0.0000657
- **测试集MSE**: 0.0003571
- **训练集R²**: 0.9645
- **测试集R²**: -0.4833
- **分析**: 训练集表现非常好，但测试集上表现差，说明模型过拟合。

#### Decision Tree
- **训练集MSE**: 0.0000746
- **测试集MSE**: 0.0008428
- **训练集R²**: 0.9597
- **测试集R²**: -2.5007
- **分析**: 训练集表现非常好，但测试集上表现差，模型明显过拟合。

#### Random Forest
- **训练集MSE**: 0.0000637
- **测试集MSE**: 0.0007482
- **训练集R²**: 0.9656
- **测试集R²**: -2.1080
- **分析**: 训练集表现非常好，但测试集上表现差，说明模型过拟合。

#### GBDT
- **训练集MSE**: 0.0000194
- **测试集MSE**: 0.0004338
- **训练集R²**: 0.9895
- **测试集R²**: -0.8020
- **分析**: 训练集表现非常好，但测试集上表现较差，说明模型过拟合。

### 综合分析
1. **Ridge Regression** 在测试集上的表现最好，R²为正，MSE相对较低，说明正则化有效地减少了过拟合。
2. **Lasso Regression** 和 **GBDT** 次之，它们在训练集和测试集上的表现都有一定差距，但仍能提供一些有用的信息。
3. **Linear Regression**、**SVR**、**Decision Tree** 和 **Random Forest** 明显存在过拟合现象，尽管在训练集上表现良好，但在测试集上的表现不佳。

### 建议
- **Ridge Regression** 是当前数据集上最好的选择，正则化有效减少了过拟合。
- 可以进一步调整正则化参数，或结合其他方法如数据增强、特征选择等，进一步优化模型性能。

### 改进代码
- 可以增加更多的特征选择和数据预处理步骤，以进一步提高模型性能。
- 采用更多的模型调优方法如随机搜索（RandomizedSearchCV）来寻找最佳参数组合。

希望这些分析和建议对您有帮助！如果需要进一步的调整或有其他问题，请告诉我。