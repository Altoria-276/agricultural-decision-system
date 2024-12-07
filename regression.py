from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from matplotlib import rcParams
import matplotlib
import seaborn as sns
import os
from scipy.optimize import curve_fit

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils import get_temp_image_path

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class RegressionModel:
    def __init__(
        self,
        model: str,
        data: pd.DataFrame,
        feature: List[str],
        target: str,
        test_size: float = 0.2,
        random_state: int | None = None,
    ):
        """
        初始化回归模型。

        Args:
            model (str): 模型类型字符串
            data (pd.DataFrame): 数据集
            feature (List[str]): 特征列表
            target (str): 目标特征
            test_size (float, optional): 测试集占数据集比例. Defaults to 0.2.
            random_state (int | None, optional): 随机数种子. Defaults to None.
        """
        self.data = data
        self.feature = feature
        self.target = target
        self.X = data[feature]
        self.y = data[target]
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.model = self.__load_model(model)
        self.is_trained = False
        self.shap_values = None

    def __load_model(self, model: str):
        if model == "Ridge":
            return Ridge()
        elif model == "Lasso":
            return Lasso()
        elif model == "SVR":
            return SVR()
        elif model == "DecisionTree":
            return DecisionTreeRegressor()
        elif model == "RandomForest":
            return RandomForestRegressor()
        else:
            return LinearRegression()

    def train_and_evaluate_model(self):
        """
        训练回归模型并评估其性能。

        返回：
        - 包含 MAE & RMSE & R² 的字典。
        """
        self.shap_values = None

        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # 训练模型
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        # 进行预测
        X_test_scaled = self.scaler.transform(X_test)
        y_test_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)

        # 计算评估指标
        rmse = root_mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # 存储结果
        self.results = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "y_train": y_train,
            "y_train_pred": y_train_pred,
            "y_test": y_test,
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

            test_data_scaled = self.scaler.transform(test_data)
            predicted_value = self.model.predict(test_data_scaled)[0]

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
        curve_data_scaled = self.scaler.transform(curve_data)
        y_pred_curve = self.model.predict(curve_data_scaled)

        # 绘制曲线

        fig, ax = plt.subplots()

        ax.plot(variable_range, y_pred_curve, label=f"预测曲线 ({variable_feature})", color="blue", lw=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)  # 零基线
        ax.set_xlabel(f"{variable_feature} 值", fontsize=14)
        ax.set_ylabel("水稻 Cd 预测值", fontsize=14)
        ax.set_title(f"水稻 Cd 预测值随 {variable_feature} 的变化", fontsize=16)

        ax.grid(alpha=0.5)
        ax.legend(fontsize=12)

        file_path = os.path.join(get_temp_image_path(), "img_prediction_curve.png")
        fig.savefig(file_path)
        plt.close()

        return file_path

    def plot_coefficients(self):
        """
        使用反事实推理的方法计算每个特征的权重，并绘制饼图。
        """
        # 原始预测值
        X_scaled = self.scaler.transform(self.X)
        original_pred = self.model.predict(X_scaled)

        # 初始化权重列表
        counterfactual_weights = []

        # 遍历每个特征
        for feature in self.feature:
            # 创建副本并将当前特征设为 0
            X_counterfactual = self.X.copy()
            X_counterfactual[feature] = 0

            # 预测新值
            X_counterfactual_scaled = self.scaler.transform(X_counterfactual)
            counterfactual_pred = self.model.predict(X_counterfactual_scaled)

            # 计算原始预测与反事实预测的绝对差值的均值
            weight = np.mean(np.abs(original_pred - counterfactual_pred))
            counterfactual_weights.append(weight)

        # 归一化权重，使其总和为 1
        total_weight = sum(counterfactual_weights)
        normalized_weights = [w / total_weight for w in counterfactual_weights]
        """
        绘制模型系数的绝对值饼图。
        """

        fig, ax = plt.subplots()

        ax.pie(normalized_weights, labels=self.feature, autopct="%1.1f%%", startangle=90)
        ax.set_title("基于系数的特征重要性")
        ax.axis("equal")

        img_path = os.path.join(get_temp_image_path(), "coefficients.png")
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

        img_path = os.path.join(get_temp_image_path(), "fitting_effect.png")
        fig.savefig(img_path)
        plt.close()
        return img_path

    def plot_shap_importance(self):
        """
        绘制基于 SHAP 值的特征重要性图。
        """
        if not self.shap_values:
            X_scaled = self.scaler.transform(self.X)
            explainer = shap.Explainer(self.model.predict, X_scaled)
            self.shap_values = explainer(X_scaled)

        fig, ax = plt.subplots(figsize=(12, 10))

        shap.summary_plot(self.shap_values, X_scaled, plot_type="bar", show=False)
        ax.set_title("基于 SHAP 的特征重要性分析")
        ax.set_xlabel("特征重要性")
        ax.set_ylabel("特征")

        img_path = os.path.join(get_temp_image_path(), "shap_importance.png")
        fig.savefig(img_path)
        plt.close()
        return img_path

    def get_top_feature(self):
        """
        获取前 5 个最重要的特征。
        """
        if not self.shap_values:
            X_scaled = self.scaler.transform(self.X)
            explainer = shap.Explainer(self.model.predict, X_scaled)
            self.shap_values = explainer(X_scaled)

        mean_abs_shap = np.abs(self.shap_values.values).mean(0)  # 对所有样本取平均
        feature_importance = list(zip(self.X.columns, mean_abs_shap, self.X.mean()))
        sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        top_6_features = [[feature[0], feature[2]] for feature in sorted_feature_importance[:6]]

        return top_6_features

    def plot_hessian_matrix(self):
        """
        计算并绘制上三角的哈森矩阵。
        """
        # 计算特征的平均值
        X_mean = self.X.mean(axis=0)

        # 使用平均特征值进行基准预测
        X_mean = X_mean.values.reshape(1, -1)
        X_mean = pd.DataFrame(X_mean, columns=self.X.columns)  # 确保 X_mean 是 DataFrame 类型
        X_mean_scaled = self.scaler.transform(X_mean)
        y_base = self.model.predict(X_mean_scaled)

        # 初始化哈森矩阵
        hessian_matrix = np.zeros((len(self.feature), len(self.feature)))

        # 计算哈森矩阵中的每个元素
        for i in range(len(self.feature)):
            for j in range(i, len(self.feature)):  # 只计算上三角部分
                # 复制原始数据集，并确保保持列名
                X_copy = self.X.copy()

                # 将第 i 和第 j 个特征的值分别乘以 1.01
                X_copy[self.feature[i]] = X_copy[self.feature[i]] * 1.01
                X_copy[self.feature[j]] = X_copy[self.feature[j]] * 1.01

                # 确保传入的数据保持 DataFrame 类型
                X_copy = pd.DataFrame(X_copy, columns=self.X.columns)

                # 进行预测
                X_copy_scaled = self.scaler.transform(X_copy)
                y1 = self.model.predict(X_copy_scaled)

                # 计算差异
                diff_y = y1 - y_base
                diff_X_i = X_copy[self.feature[i]] - self.X[self.feature[i]]
                diff_X_j = X_copy[self.feature[j]] - self.X[self.feature[j]]

                # 计算哈森矩阵的值
                hessian_matrix[i, j] = np.sum((diff_y**2) / (diff_X_i * diff_X_j))

                # 对称填充
                if i != j:
                    hessian_matrix[j, i] = hessian_matrix[i, j]

        # 使用 seaborn 绘制热图
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            hessian_matrix,
            annot=False,
            fmt=".4f",
            xticklabels=self.feature,
            yticklabels=self.feature,
            cmap="coolwarm",
            center=0,
            mask=np.tril(np.ones_like(hessian_matrix, dtype=bool)),
            cbar_kws={"label": "相互影响强度"},
            ax=ax,
        )

        # 添加标题和标签
        ax.set_title("关联辅因分析", fontsize=18, weight="bold")
        ax.set_xlabel("特征", fontsize=14, weight="bold")
        ax.set_ylabel("特征", fontsize=14, weight="bold")

        # 优化图形边框和网格
        ax.grid(False)
        fig.tight_layout()

        img_path = os.path.join(get_temp_image_path(), "hessian.png")
        fig.savefig(img_path)
        plt.close()
        return img_path

    # TODO 将self.X替换成固定值
    def plot_overrate(self, threshold_range=None, y_threshold=0.2):
        """
        绘制超标率与阈值的关系，并拟合曲线
        Args:
            threshold_range (tuple): 阈值范围，默认使用自定义范围。
            y_threshold (float): 第二轮筛选y值的阈值，默认为0.2。
        """

        # 如果没有提供阈值范围，则自动生成一个合理的范围
        if threshold_range is None:
            threshold_range = (self.data['Cd'].min(), self.data['Cd'].max())

        # 生成多个阈值并计算对应的超标率
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
        overrate = []

        # 获取用于拟合的真实数据 x_data 和 y_data
        for threshold in thresholds:
            # 第一轮筛选：Cd > threshold
            selected_data = self.data[self.data['Cd'] > threshold]

            # 第二轮筛选：预测值y > y_threshold
            predict_y = self.model.predict(selected_data[self.feature])
            selected_data['predicted_y'] = predict_y
            overrate_samples = selected_data[selected_data['predicted_y'] > y_threshold]

            # 计算超标率
            overrate.append(len(overrate_samples) / len(self.data))

        # 用于拟合的真实数据
        x_data = thresholds
        y_data = overrate

        # 拟合曲线函数
        def fit_func(x, theta, lambda_val):
            return (1 / (1 + x ** lambda_val)) ** theta

        # 使用curve_fit进行拟合，获得λ和θ的最优值
        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=[1, 1])

        # 打印拟合的参数
        print("Optimal parameters (theta, lambda): ", popt)
        print("Covariance of parameters: ", pcov)

        # 使用拟合参数绘图
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = fit_func(x_fit, *popt)

        # 绘制超标率与阈值的关系图
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, label='超标率', color='blue')
        plt.plot(x_fit, y_fit, 'r-', label=f'拟合曲线: f(x) = (1/(1+x^{popt[1]}))^{popt[0]}', linestyle='--')
        plt.xlabel('Cd阈值')
        plt.ylabel('超标率')
        plt.title('超标率与Cd阈值的关系')
        plt.legend()
        plt.grid(True)

        # 用于绘制拟合曲线的阈值范围
        x2_data = np.linspace(0, 10, 100)
        y2_data = fit_func(x2_data, *popt)

        plt.figure(figsize=(8, 6))
        plt.plot(x2_data, y2_data, 'g-', label='拟合曲线用于绘图')
        plt.xlabel('Cd阈值范围')
        plt.ylabel('拟合的超标率')
        plt.title('拟合曲线图')
        plt.legend()
        plt.grid(True)

        # TODO 先打游戏去了，反求y明天再做

