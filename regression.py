import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from matplotlib import rcParams
import matplotlib

# 设置字体为 macOS 系统自带的中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang HK']  # 或者 'PingFang SC'
matplotlib.rcParams['axes.unicode_minus'] = False


def train_and_evaluate_model(model, X, y, test_size=0.2, random_state=42):
    """
    训练模型并评估性能。

    参数:
    - model: sklearn 的回归模型对象（例如 Ridge、Lasso 等）。
    - X: 特征数据集。
    - y: 目标数据集。
    - test_size: 测试集比例，默认 0.2。
    - random_state: 随机数种子，默认 42。

    返回:
    - coefficients: 模型的系数（权重）。
    - mse: 均方误差。
    - r2: 决定系数。
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 模型训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 输出结果
    print(f"均方误差 (MSE): {mse}")
    print(f"决定系数 (R²): {r2}")

    # 打印模型系数
    print("\n模型的系数:")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print(coefficients)

    # 返回模型系数、MSE 和 R²
    return model.coef_, mse, r2


def plot_coefficients(coefficients, feature_names):
    """
    根据模型的系数绘制饼图。

    参数:
    - coefficients: 模型的系数。
    - feature_names: 特征名称列表。
    """
    # 计算绝对值权重
    abs_coefficients = abs(coefficients)

    # 绘制饼图
    plt.figure(figsize=(10, 8))
    plt.pie(abs_coefficients, labels=feature_names, autopct='%1.1f%%', startangle=90)
    plt.title('特征权重的绝对值饼图')
    plt.axis('equal')  # 保持饼图为圆形
    plt.show()


# 加载数据
file_path = '水稻点位148.xlsx'
data = pd.ExcelFile(file_path)
raw_data = data.parse('原始数据')

# 定义特征和目标
features = ['P', 'K', 'N', 'Cr', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'Se', 'Mo',
            'Na', 'Al', 'Si', 'Ca', 'Fe', 'Hg', 'La', 'Mg', 'Mn', '有效态Cd']
target = '水稻Cd'

X = raw_data[features]
y = raw_data[target]

# 使用 Ridge 模型训练
ridge_model = Ridge()
coefficients, mse, r2 = train_and_evaluate_model(ridge_model, X, y)

# 绘制系数饼图
plot_coefficients(coefficients, features)
