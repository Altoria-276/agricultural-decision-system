from typing import List
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
        return self.results

    def plot_coefficients(self):
        """
        绘制模型系数的绝对值饼图。
        """
        coefficients = abs(self.results["coefficients"]["Coefficient"])
        plt.figure(figsize=(10, 8))
        plt.pie(coefficients, labels=self.feature, autopct="%1.1f%%", startangle=90)
        plt.title("基于系数的特征重要性")
        plt.axis("equal")

        img_path = os.path.join(".", "Images", "coefficients.png")
        plt.savefig(img_path)
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

        img_path = os.path.join(".", "Images", "fitting_effect.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

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

        img_path = os.path.join(".", "Images", "shap_importance.png")
        plt.savefig(img_path)
        plt.close()
        return img_path


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

    # 创建实例
    model = RegressionModel(
        model="Ridge", data=pd.read_excel(os.path.join(".", "数据", "水稻点位148.xlsx"), "原始数据"), features=features, target=target
    )

    # 训练并评估模型
    results = model.train_and_evaluate_model()

    # 绘制结果
    model.plot_coefficients()
    model.plot_fitting_effect()
    model.plot_shap_importance()
