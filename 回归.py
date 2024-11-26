# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
import shap
import numpy as np
import matplotlib

# 设置 matplotlib 参数以便更好地可视化
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK"]
matplotlib.rcParams["axes.unicode_minus"] = False


class RegressionPipeline:
    def __init__(self, model, features, target, test_size=0.2, random_state=42):
        """
        初始化回归管道。

        参数：
        - model: 回归模型（例如 Ridge、Lasso）。
        - features: 特征列名称的列表。
        - target: 目标列名称。
        - test_size: 测试集占数据集的比例。
        - random_state: 随机种子以确保可重复性。
        """
        self.model = model
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self, file_path, sheet_name):
        """
        从 Excel 文件加载数据。

        参数：
        - file_path: Excel 文件的路径。
        - sheet_name: 要加载的工作表名称。
        """
        data = pd.ExcelFile(file_path)
        self.data = data.parse(sheet_name)
        self.X = self.data[self.features]
        self.y = self.data[self.target]

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
            "coefficients": pd.DataFrame({"Feature": self.features, "Coefficient": self.model.coef_}),
            "mse": mse,
            "r2": r2,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
        }
        return self.results

    def plot_coefficients(self):
        """
        绘制模型系数的绝对值饼图。
        """
        coefficients = abs(self.results["coefficients"]["Coefficient"])
        plt.figure(figsize=(10, 8))
        plt.pie(coefficients, labels=self.features, autopct="%1.1f%%", startangle=90)
        plt.title("基于系数的特征重要性")
        plt.axis("equal")
        plt.show()

    def plot_fitting_effect(self):
        """
        绘制训练集和测试集的拟合效果图。
        """
        y_train = self.results["y_train"]
        y_train_pred = self.results["y_train_pred"]
        y_test = self.results["y_test"]
        y_test_pred = self.results["y_test_pred"]

        plt.figure(figsize=(14, 6))

        # 绘制训练数据
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, edgecolors="k", facecolors="none", label="训练数据")
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title("训练数据")
        plt.legend()

        # 绘制测试数据
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, edgecolors="k", facecolors="none", label="测试数据")
        reg = LinearRegression().fit(y_test.values.reshape(-1, 1), y_test_pred)
        plt.plot(y_test, reg.predict(y_test.values.reshape(-1, 1)), "b-", lw=2)
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title("测试数据")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_shap_importance(self):
        """
        绘制基于 SHAP 值的特征重要性图。
        """
        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)
        shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
        plt.title("基于 SHAP 的特征重要性分析")
        plt.xlabel("特征重要性")
        plt.ylabel("特征")
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 定义特征和目标
    features = [
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

    # 创建管道实例
    pipeline = RegressionPipeline(model=Ridge(), features=features, target=target)

    # 加载数据
    pipeline.load_data(file_path="水稻点位148.xlsx", sheet_name="原始数据")

    # 训练并评估模型
    results = pipeline.train_and_evaluate_model()

    # 绘制结果
    pipeline.plot_coefficients()
    pipeline.plot_fitting_effect()
    pipeline.plot_shap_importance()
