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
import seaborn as sns

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
    image_path: str,
    low_threshold: float = 0.2,
    high_threshold: float = 0.9,
    beta: int = 10,
    min_size: int = 100,
    dpi: int = 400,
):
    """
    对图像进行随机游走分割、去除小区域以及骨架提取，并保存结果图像
    :param image_path: str, 输入图像的路径。
    :param low_threshold: float, 分割时的低阈值。
    :param high_threshold: float, 分割时的高阈值。
    :param beta: int, 随机游走算法的边缘敏感度参数。
    :param min_size: int, 去除小区域的最小面积。
    :param dpi: int, 图像的DPI。
    :return: str, 保存图像的路径。
    """
    # 加载图像
    img: np.ndarray = img_as_float(io.imread(image_path))

    # 随机游走分割
    markers: np.ndarray = np.zeros_like(img)
    markers[img > high_threshold] = 1
    markers[img < low_threshold] = 2
    seg_result = random_walker(img, markers, beta=beta, mode="bf")

    # 区域处理，去除最小区域
    label_img = label(seg_result)
    filtered_regions = morphology.remove_small_objects(label_img, min_size)

    # 二值化处理
    prop_bw = filtered_regions
    prop_bw[prop_bw == 1] = 0
    prop_bw[prop_bw >= 1] = 1

    # 骨架化
    skeleton = morphology.skeletonize(prop_bw)

    # 显示并保存结果
    fig, ax = plt.subplots(dpi=dpi)

    k = 0.8
    ax.imshow(img * k + np.double(skeleton) * (1 - k))
    ax.axis("off")
    ax.set_title("Processed (Skeleton Overlay)", fontsize=12)

    file_path = os.path.join(".", "Images", "img_random_walk.png")

    fig.savefig(file_path, dpi=dpi)
    plt.close(fig)

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

    # 输出主成分的方差贡献率
    # print(f"主成分的方差贡献率: {pca.explained_variance_ratio_}")
    # print(f"累计方差贡献率: {np.cumsum(pca.explained_variance_ratio_)}")

    # 可视化降维后的数据 (前两个主成分)
    # if n_components >= 2:
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    #     plt.title("PCA - 2D Projection")
    #     plt.xlabel("主成分1")
    #     plt.ylabel("主成分2")
    #     plt.show()

    # 可视化负载矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=params_name, columns=[f"主成分{i+1}" for i in range(n_components)])

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA负载矩阵")
    file_path = os.path.join(".", "Images", "Img_pca_loading.png")
    plt.savefig(file_path)

    return file_path


def img_pie_percent(data: pd.DataFrame, label: str = "2012年"):
    offset = data[label] - data["基准"]
    # 定义区间
    bins = [float("-inf"), 0, 0.01, 0.05, 0.1, 0.2, float("inf")]
    labels = ["<0%", "0-1%", "1-5%", "5-10%", "10-20%", ">20%"]

    # 将数据分类到区间
    categorized_data = pd.cut(offset, bins=bins, labels=labels, include_lowest=True)

    # 计算每个区间的占比
    value_counts = categorized_data.value_counts(normalize=True) * 100

    # 绘制饼状图
    plt.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Percentage Distribution of Data")
    file_path = os.path.join(".", "Images", "img_pie_percent.png")
    plt.savefig(file_path)

    return file_path


def img_line_percent(data: pd.DataFrame):
    se = data.mean()[3:].pct_change()[1:] * 100
    # 绘制折线图
    se.plot(kind="line", marker="o", linestyle="-", color="blue")

    # 添加标题和标签
    plt.title("年度百分比变化")
    plt.xlabel("年份")
    plt.ylabel("百分比变化(%)")
    # 显示网格
    plt.grid(True)

    file_path = os.path.join(".", "Images", "img_line_percent.png")
    plt.savefig(file_path)


if __name__ == "__main__":
    image_path = "Images/img2.png"  # 文件可更换
    file_path = "Images/processed_images/img2.png"  # 路径可更换
    result = img_random_walk_process(image_path, file_path)

    # 示例调用
    file_path = "数据/水稻点位148.xlsx"  # Excel文件路径
    sheet_name = "原始数据"  # Excel表单名称
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
    ]

    # 读取Excel数据
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # 执行 PCA 并降到8个主成分
    pca_result, loadings_df = img_pca_loading(data, params_name, n_components=8)
