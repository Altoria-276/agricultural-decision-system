import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib


# 设置字体为 macOS 系统自带的中文字体
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK"]
matplotlib.rcParams["axes.unicode_minus"] = False


class SoilCdModel:
    """
    土壤 Cd 与水稻 Cd 关系建模类
    """

    def __init__(self, features, target, model=Ridge()):
        """
        初始化模型类
        参数:
        - features: 特征列名称列表
        - target: 目标列名称
        - model: 使用的回归模型（默认 Ridge 回归）
        """
        self.features = features
        self.target = target
        self.model = model
        self.trained_model = None

    def train_model(self, data, test_size=0.2, random_state=42):
        """
        训练模型
        参数:
        - data: 输入的数据集
        - test_size: 测试集比例（默认 0.2）
        - random_state: 随机种子（默认 42）
        返回:
        - 训练好的模型对象
        """
        X = data[self.features]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"训练完成: 均方误差 (MSE): {mse}, 决定系数 (R²): {r2}")
        self.trained_model = self.model
        return self.model, mse, r2

    def calculate_threshold(self, fixed_values, variable_feature, target_value, precision=0.001):
        """
        计算给定水稻 Cd 值时对应的土壤 Cd 值
        参数:
        - fixed_values: 固定特征值（字典形式）
        - variable_feature: 浮动的特征名称
        - target_value: 目标值（水稻 Cd）
        - precision: 计算精度（默认 0.001）
        返回:
        - 对应的土壤 Cd 值
        """
        if not self.trained_model:
            raise RuntimeError("模型尚未训练，请先调用 train_model 方法。")
        lower_bound, upper_bound = 0, 2  # 假设土壤 Cd 的范围
        while upper_bound - lower_bound > precision:
            mid_point = (lower_bound + upper_bound) / 2
            test_values = fixed_values.copy()
            test_values[variable_feature] = mid_point
            test_data = pd.DataFrame([test_values])[self.features]
            predicted_value = self.trained_model.predict(test_data)[0]
            if predicted_value < target_value:
                lower_bound = mid_point
            else:
                upper_bound = mid_point
        return (lower_bound + upper_bound) / 2

    def plot_prediction_curve(self, fixed_values, variable_feature, variable_range, threshold_value=None, threshold_label=None):
        """
        绘制预测曲线
        参数:
        - fixed_values: 固定特征值（字典形式）
        - variable_feature: 浮动的特征名称
        - variable_range: 浮动特征的变化范围
        - threshold_value: 阈值（可选，用于标注安全值）
        - threshold_label: 阈值标注的文本说明（可选）
        """
        if not self.trained_model:
            raise RuntimeError("模型尚未训练，请先调用 train_model 方法。")

        # 构造预测数据
        curve_data = pd.DataFrame(fixed_values, index=range(len(variable_range)))
        curve_data[variable_feature] = variable_range

        curve_data = curve_data[self.features]  # 确保特征顺序一致

        # 预测值
        y_pred_curve = self.trained_model.predict(curve_data)

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(variable_range, y_pred_curve, label=f"预测曲线 ({variable_feature})", color="blue", lw=2)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.7)  # 零基线
        plt.xlabel(f"{variable_feature} 值", fontsize=14)
        plt.ylabel("水稻 Cd 预测值", fontsize=14)
        plt.title(f"水稻 Cd 预测值随 {variable_feature} 的变化", fontsize=16)

        plt.grid(alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


# ============================
# 主程序入口
# ============================

# 加载数据
file_path = "水稻点位148.xlsx"
data = pd.ExcelFile(file_path)
raw_data = data.parse("原始数据")

# 定义特征和目标
features = ["Cd", "Pb", "Al", "Ca", "Mn"]
target = "水稻Cd"

# 初始化模型类
soil_model = SoilCdModel(features=features, target=target)

# 训练模型
soil_model.train_model(raw_data)

# 固定特征值
base_feature_values = {"Pb": 43.4, "Al": 10.8, "Ca": 747.2, "Mn": 240.2}

# 计算土壤 Cd 的安全阈值
threshold_cd = soil_model.calculate_threshold(fixed_values=base_feature_values, variable_feature="Cd", target_value=0.2)
print(f"当水稻 Cd 为 0.2 时，对应的土壤 Cd 安全阈值为: {threshold_cd:.4f}")

variable_range = np.linspace(0, 2, 100)
soil_model.plot_prediction_curve(
    fixed_values=base_feature_values,
    variable_feature="Cd",
    variable_range=variable_range,
    threshold_value=threshold_cd,
    threshold_label=f"安全阈值 (Cd = {threshold_cd:.4f})",
)
