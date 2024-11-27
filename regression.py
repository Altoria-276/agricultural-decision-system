from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib import rcParams
import matplotlib
import os

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class RegressionModel:
    def __init__(self, model: str, data: pd.DataFrame, feature: List[str], target: str, test_size: float = 0.2, random_state: int = 42):
        self.data = data
        self.feature = feature
        self.target = target
        self.X = data[feature]
        self.y = data[target]
        self.test_size = test_size
        self.random_state = random_state
        self.model = self.__load_model(model)
        self.is_trained = False

    def __load_model(self, model: str):
        if model == "Ridge":
            return Ridge()
        else:
            return Ridge()

    def train_and_evaluate_model(self):
        """
        训练回归模型并评估其性能。

        返回：
        - 包含模型系数、MSE、R² 和预测结果的字典。
        """
        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # 训练模型
        self.model.fit(X_train, y_train)

        # 进行预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # 存储结果
        self.results = {
            "coefficients": pd.DataFrame({"Feature": self.feature, "Coefficient": self.model.coef_}),
            "mse": mse,
            "r2": r2,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
        }

        self.is_trained = True

        return self.results

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
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        lower_bound, upper_bound = 0, 2  # 假设土壤 Cd 的范围
        while upper_bound - lower_bound > precision:
            mid_point = (lower_bound + upper_bound) / 2
            test_values = fixed_values.copy()
            test_values[variable_feature] = mid_point
            test_data = pd.DataFrame([test_values])[self.feature]
            predicted_value = self.model.predict(test_data)[0]
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
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        # 构造预测数据
        curve_data = pd.DataFrame(fixed_values, index=range(len(variable_range)))
        curve_data[variable_feature] = variable_range

        curve_data = curve_data[self.feature]  # 确保特征顺序一致

        # 预测值
        y_pred_curve = self.model.predict(curve_data)

        # 绘制曲线

        fig, ax = plt.subplots()

        ax.plot(variable_range, y_pred_curve, label=f"预测曲线 ({variable_feature})", color="blue", lw=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)  # 零基线
        ax.set_xlabel(f"{variable_feature} 值", fontsize=14)
        ax.set_ylabel("水稻 Cd 预测值", fontsize=14)
        ax.set_title(f"水稻 Cd 预测值随 {variable_feature} 的变化", fontsize=16)

        ax.grid(alpha=0.5)
        ax.legend(fontsize=12)

        file_path = os.path.join(".", "Images", "img_prediction_curve.png")
        fig.savefig(file_path)
        plt.close()

        return file_path

    def plot_coefficients(self):
        """
        绘制模型系数的绝对值饼图。
        """

        fig, ax = plt.subplots()

        coefficients = abs(self.results["coefficients"]["Coefficient"])
        ax.pie(coefficients, labels=self.feature, autopct="%1.1f%%", startangle=90)
        ax.set_title("基于系数的特征重要性")
        ax.axis("equal")

        img_path = os.path.join(".", "Images", "coefficients.png")
        fig.savefig(img_path)
        plt.close()
        return img_path

    def plot_fitting_effect(self):
        """
        绘制训练集和测试集的拟合效果图。
        """
        y_train = self.results["y_train"]
        y_train_pred = self.results["y_train_pred"]
        y_test = self.results["y_test"]
        y_test_pred = self.results["y_test_pred"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # 绘制训练数据
        ax1.scatter(y_train, y_train_pred, edgecolors="k", facecolors="none", label="训练数据")
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
        ax1.set_xlabel("实际值")
        ax1.set_ylabel("预测值")
        ax1.set_title("训练数据")
        ax1.legend()

        # 绘制测试数据
        ax2.scatter(y_test, y_test_pred, edgecolors="k", facecolors="none", label="测试数据")
        reg = LinearRegression().fit(y_test.values.reshape(-1, 1), y_test_pred)
        ax2.plot(y_test, reg.predict(y_test.values.reshape(-1, 1)), "b-", lw=2)
        ax2.set_xlabel("实际值")
        ax2.set_ylabel("预测值")
        ax2.set_title("测试数据")
        ax2.legend()

        fig.tight_layout()

        img_path = os.path.join(".", "Images", "fitting_effect.png")
        fig.savefig(img_path)
        plt.close()
        return img_path

    def plot_shap_importance(self):
        """
        绘制基于 SHAP 值的特征重要性图。
        """
        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)

        fig, ax = plt.subplots(figsize=(10, 8))

        shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
        ax.set_title("基于 SHAP 的特征重要性分析")
        ax.set_xlabel("特征重要性")
        ax.set_ylabel("特征")

        img_path = os.path.join(".", "Images", "shap_importance.png")
        fig.savefig(img_path)
        plt.close()
        return img_path


# 示例用法
if __name__ == "__main__":
    # 定义特征和目标
    feature = [
        "P",
        "K",
        "N",
        "Cr",
        "Cu",
        "Zn",
        "As",
        "Cd",
        "Pb",
        "Se",
        "Mo",
        "Na",
        "Al",
        "Si",
        "Ca",
        "Fe",
        "Hg",
        "La",
        "Mg",
        "Mn",
        "有效态Cd",
    ]
    target = "水稻Cd"

    # 创建实例
    model = RegressionModel(
        model="Ridge",
        data=pd.read_excel(os.path.join(".", "数据", "水稻点位148.xlsx"), "原始数据"),
        feature=feature,
        target=target,
    )

    # 训练并评估模型
    results = model.train_and_evaluate_model()

    # 绘制结果
    model.plot_coefficients()
    model.plot_fitting_effect()
    model.plot_shap_importance()

    base_feature_values = {"Pb": 43.4, "Al": 10.8, "Ca": 747.2, "Mn": 240.2}

    # 计算土壤 Cd 的安全阈值
    threshold_cd = model.calculate_threshold(fixed_values=base_feature_values, variable_feature="Cd", target_value=0.2)
    print(f"当水稻 Cd 为 0.2 时，对应的土壤 Cd 安全阈值为: {threshold_cd:.4f}")

    variable_range = np.linspace(0, 2, 100)
    model.plot_prediction_curve(
        fixed_values=base_feature_values,
        variable_feature="Cd",
        variable_range=variable_range,
        threshold_value=threshold_cd,
        threshold_label=f"安全阈值 (Cd = {threshold_cd:.4f})",
    )
