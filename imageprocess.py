import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_float, measure, morphology, io
from skimage.segmentation import random_walker
from skimage.measure import label, regionprops
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import seaborn as sns
from utils import get_temp_image_path

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

params_name = [
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
    "水稻Cd",
]


def img_random_walk_process(
    image_path_color: str,
    image_path_grey: str,
    low_threshold_precent: float = 20,
    high_threshold_precent: float = 80,
    beta: int = 10,
    min_size: int = 100,
    dpi: int = 400,
):
    """
    对图像进行随机游走分割、去除小区域以及骨架提取，并保存结果图像
    :param image_path_color: str, 输入图像的路径。
    :param image_path_grey: str, 输入灰度图的路径。
    :param low_threshold_percent: float, 分割时的低阈值。
    :param high_threshold_percent: float, 分割时的高阈值。
    :param beta: int, 随机游走算法的边缘敏感度参数。
    :param min_size: int, 去除小区域的最小面积。
    :return: str, 保存图像的路径。
    """
    # 加载图像
    img_color: np.ndarray = io.imread(image_path_color)
    img_grey: np.ndarray = io.imread(image_path_grey, as_gray=True)
    img_grey = np.sqrt(1 - img_grey)

    non_zero_elements = img_grey[img_grey != 0]

    low_threshold = np.percentile(non_zero_elements, low_threshold_precent)
    high_threshold = np.percentile(non_zero_elements, high_threshold_precent)

    # 随机游走分割
    markers: np.ndarray = np.zeros_like(img_grey)
    markers[img_grey > high_threshold] = 1
    markers[img_grey < low_threshold] = 2
    seg_result = random_walker(img_grey, markers, beta=beta, mode="bf")

    # 区域处理，去除最小区域
    label_img = label(seg_result)
    filtered_regions = morphology.remove_small_objects(label_img, min_size)

    # 二值化处理
    prop_bw = filtered_regions
    prop_bw[prop_bw == 1] = 0
    prop_bw[prop_bw >= 1] = 1

    # 骨架化
    skeleton = morphology.skeletonize(prop_bw)

    img_color[skeleton, :3] = [255, 0, 0]

    # 显示并保存结果
    fig, ax = plt.subplots()

    ax.imshow(img_color)
    ax.axis("off")
    ax.set_title("污染路径图", fontsize=12)

    file_path = os.path.join(get_temp_image_path(), "img_random_walk.png")

    fig.savefig(file_path)
    plt.close()

    return file_path


def img_pca_loading(data: pd.DataFrame, params_name: List[str] = params_name, n_components: int = 8):
    # 选择需要进行PCA分析的列
    data_selected = data[params_name]

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_selected)

    # 执行PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)

    # 获取负载矩阵，并转换为百分比
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=params_name, columns=[f"主成分{i+1}" for i in range(n_components)])

    # 创建图形
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 2 * n_components))  # 每个主成分一个小图
    if n_components == 1:  # 如果只有一个主成分，axes 是单个对象，而不是数组
        axes = [axes]

    # 绘制每个主成分的负载矩阵
    for i in range(n_components):
        ax = axes[i]

        # 获取当前主成分的负载矩阵
        loading_values = loadings_df.iloc[:, i]
        loading_percent = loading_values  # 直接使用负载值，而不转换为百分比

        # 绘制负载矩阵条形图，横轴是影响成分，纵轴是负载值
        ax.bar(loading_percent.index, loading_percent, color="blue")

        # 设置纵轴范围固定在 -1 到 1
        ax.set_ylim(-1, 1)

        # 设置图形标题与标签
        ax.set_title(f"主成分 {i + 1}", fontsize=12)
        ax.set_xlabel("成分", fontsize=10)
        ax.set_ylabel("负载值", fontsize=10)

        # 标注负载值（没有百分号）
        for j, value in enumerate(loading_percent):
            ax.text(j, value, f"{value:.2f}", ha="center", va="bottom" if value > 0 else "top", fontsize=9)

    # 保存图像
    file_path = os.path.join(get_temp_image_path(), "Img_pca_loading_components.png")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close()

    return file_path


def img_pie_percent(data: pd.DataFrame, label_basic: str = "2008年", label: str = "2012年"):
    percent = (data[label] - data[label_basic]) / data[label_basic]
    # 定义区间
    bins = [float("-inf"), 0, 0.01, 0.05, 0.1, 0.2, float("inf")]
    labels = ["<0%", "0-1%", "1-5%", "5-10%", "10-20%", ">20%"]

    # 将数据分类到区间
    categorized_data = pd.cut(percent, bins=bins, labels=labels, include_lowest=True)

    # 计算每个区间的占比
    value_counts = categorized_data.value_counts(normalize=True) * 100

    fig, ax = plt.subplots()

    # 绘制饼状图
    ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("数据百分比变化占比")
    file_path = os.path.join(get_temp_image_path(), "img_pie_percent.png")
    fig.savefig(file_path)
    plt.close()

    return file_path


def img_line_percent(data: pd.DataFrame, start_time: str = "2008年", end_time: str = "2020年"):
    fig, ax = plt.subplots(figsize=(15, 6))
    ans = data[[item for item in data.columns if item.endswith("年") and start_time <= item <= end_time]].T.pct_change().T.iloc[:, 1:]
    # 绘制箱线图
    ax.boxplot(x=ans)

    # 添加标题和标签
    ax.set_title("年度百分比变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("百分比变化(%)")
    ax.set_xticks(range(1, len(ans.columns) + 1), ans.columns)

    file_path = os.path.join(get_temp_image_path(), "img_line_percent.png")
    fig.savefig(file_path)
    plt.close()

    return file_path


def plot_anova(data, params_name):
    """
    对给定的数据集中的每一列因素进行单因素方差分析，并绘制每个因素的p值柱状图。
    参数:
        data (pd.DataFrame): 包含待分析数据的DataFrame。
        params_name (list of str): 需要进行方差分析的因素名称列表。
    """
    # 创建一个虚拟的二元组别
    data["group"] = np.random.choice(["A", "B"], size=len(data), p=[0.5, 0.5])

    # 创建一个空字典来存储每个因素的p值
    p_values = {}

    # 对每个因素执行ANOVA
    for param in params_name:
        group_a = data[data["group"] == "A"][param]
        group_b = data[data["group"] == "B"][param]

        # 检查是否有足够的数据点来进行ANOVA
        if len(group_a) > 1 and len(group_b) > 1:
            f_val, p_val = stats.f_oneway(group_a, group_b)
            p_values[param] = p_val
        else:
            print(f"Warning: Not enough data points for {param} to perform ANOVA.")

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(p_values.keys(), p_values.values(), color="skyblue")

    # 在y=0.05处画一条虚线
    ax.axhline(y=0.05, color="r", linestyle="--")

    # 设置图表标题和轴标签
    ax.set_title("P-values from ANOVA for Each Factor")
    ax.set_xlabel("Factors")
    ax.set_ylabel("P-value")

    # 自动调整x轴标签以避免重叠
    plt.xticks(rotation=45, ha="right")

    # 保存图片
    file_path = os.path.join(get_temp_image_path(), "img_anova_p_values.png")

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close()

    return file_path


def plot_correlation_matrix(data, params_name):
    """
    对给定的数据集中的每一列因素进行两两相关性分析，并绘制相关性矩阵热图。
    参数:
        data (pd.DataFrame): 包含待分析数据的DataFrame。
        params_name (list of str): 需要进行相关性分析的因素名称列表。
    """
    # 选择需要分析的列
    selected_data = data[params_name]

    # 计算相关性矩阵
    corr_matrix = selected_data.corr()

    # 绘制相关性矩阵热图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

    # 设置图表标题
    ax.set_title("Correlation Matrix of Factors")

    # 保存图片
    file_path = os.path.join(get_temp_image_path(), "img_correlation_matrix.png")

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close()

    return file_path


def plot_Cd(data, x_threshold=None, y_threshold=None):
    x = data["Cd"]
    y = data["水稻Cd"]

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o")

    ax.set_xlabel("Cd")
    ax.set_ylabel("水稻Cd")

    # 绘制 x 轴阈值线
    if x_threshold is not None:
        ax.axvline(x=x_threshold, color="r", linestyle="--", label=f"Cd 阈值: {x_threshold}")

    # 绘制 y 轴阈值线
    if y_threshold is not None:
        ax.axhline(y=y_threshold, color="b", linestyle="--", label=f"水稻Cd 阈值: {y_threshold}")

    file_path = os.path.join(get_temp_image_path(), "img_Cd.png")
    fig.savefig(file_path)
    plt.close()

    return file_path


def plot_bar(data: pd.DataFrame, label: str):
    fig, ax = plt.subplots()
    ax.bar(x=data.index, height=data[label], color="skyblue")

    ax.set_title(f"{label} 比较")

    file_path = os.path.join(get_temp_image_path(), f"img_{label}_compare.png")
    fig.savefig(file_path)
    plt.close()

    return file_path
