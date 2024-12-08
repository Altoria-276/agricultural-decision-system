# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
import shap
import numpy as np
import matplotlib
import seaborn as sns

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
        使用反事实推理的方法计算每个特征的权重，并绘制饼图。
        """
        # 原始预测值
        original_pred = self.model.predict(self.X)

        # 初始化权重列表
        counterfactual_weights = []

        # 遍历每个特征
        for feature in self.features:
            # 创建副本并将当前特征设为 0
            X_counterfactual = self.X.copy()
            X_counterfactual[feature] = 0

            # 预测新值
            counterfactual_pred = self.model.predict(X_counterfactual)

            # 计算原始预测与反事实预测的绝对差值的均值
            weight = np.mean(np.abs(original_pred - counterfactual_pred))
            counterfactual_weights.append(weight)

        # 归一化权重，使其总和为 1
        total_weight = sum(counterfactual_weights)
        normalized_weights = [w / total_weight for w in counterfactual_weights]

        # 绘制饼图
        plt.figure(figsize=(10, 8))
        plt.pie(normalized_weights, labels=self.features, autopct="%1.1f%%", startangle=90)
        plt.title("基于反事实推理的特征权重图")
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

    def plot_hessian_matrix(self):
        """
        计算并绘制上三角的哈森矩阵。
        """
        # 计算特征的平均值
        X_mean = self.X.mean(axis=0)

        # 使用平均特征值进行基准预测
        X_mean = X_mean.values.reshape(1, -1)
        X_mean = pd.DataFrame(X_mean, columns=self.X.columns)  # 确保 X_mean 是 DataFrame 类型
        y_base = self.model.predict(X_mean)

        # 初始化哈森矩阵
        hessian_matrix = np.zeros((len(self.features), len(self.features)))

        # 计算哈森矩阵中的每个元素
        for i in range(len(self.features)):
            for j in range(i, len(self.features)):  # 只计算上三角部分
                # 复制原始数据集，并确保保持列名
                X_copy = self.X.copy()

                # 将第 i 和第 j 个特征的值分别乘以 1.01
                X_copy[self.features[i]] = X_copy[self.features[i]] * 1.01
                X_copy[self.features[j]] = X_copy[self.features[j]] * 1.01

                # 确保传入的数据保持 DataFrame 类型
                X_copy = pd.DataFrame(X_copy, columns=self.X.columns)

                # 进行预测
                y1 = self.model.predict(X_copy)

                # 计算差异
                diff_y = y1 - y_base
                diff_X_i = X_copy[self.features[i]] - self.X[self.features[i]]
                diff_X_j = X_copy[self.features[j]] - self.X[self.features[j]]

                # 计算哈森矩阵的值
                hessian_matrix[i, j] = np.sum((diff_y**2) / (diff_X_i * diff_X_j))

                # 对称填充
                if i != j:
                    hessian_matrix[j, i] = hessian_matrix[i, j]

        # 使用 seaborn 绘制热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            hessian_matrix,
            annot=False,
            fmt=".4f",
            xticklabels=self.features,
            yticklabels=self.features,
            cmap="coolwarm",
            center=0,
            mask=np.tril(np.ones_like(hessian_matrix, dtype=bool)),
            cbar_kws={"label": "相互影响强度"},
        )

        # 添加标题和标签
        plt.title("关联辅因分析", fontsize=18, weight="bold")
        plt.xlabel("特征", fontsize=14, weight="bold")
        plt.ylabel("特征", fontsize=14, weight="bold")

        # 优化图形边框和网格
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def analyze_variable_impact(self, variable):  # 新增变化影响分析
        """
        计算指定变量的变化对模型预测的影响，并绘制柱状图。

        参数：
        - variable: 要分析的变量名。
        """
        if variable not in self.features:
            raise ValueError(f"指定的变量 {variable} 不在特征列表中。")

        # 原始预测值
        original_pred = self.model.predict(self.X)
        original_mean = np.mean(original_pred)

        # 存储不同变化情况下的预测均值
        means = [original_mean]
        factors = [1.0, 0.95, 0.9, 0.8]

        for factor in factors[1:]:
            # 创建副本并调整指定变量
            X_modified = self.X.copy()
            X_modified[variable] *= factor

            # 预测新值并计算均值
            modified_pred = self.model.predict(X_modified)
            modified_mean = np.mean(modified_pred)
            means.append(modified_mean)

        # 绘制柱状图
        plt.figure(figsize=(8, 6))
        plt.bar(["原始值", "95%", "90%", "80%"], means, color=["blue", "orange", "green", "red"])
        plt.title(f"变量 {variable} 变化对预测均值的影响")
        plt.xlabel("变量调整比例")
        plt.ylabel("预测均值")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
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
    pipeline.plot_hessian_matrix()
