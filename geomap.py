import os
from typing import List, Optional
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from math import ceil, floor
from geopandas.geodataframe import GeoDataFrame
from utils import get_temp_image_path

np.random.seed(42)  # 设置种子值为42
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

space = ["东经", "北纬"]
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
params_name_2 = [
    "2007年",
    "2008年",
    "2009年",
    "2010年",
    "2011年",
    "2012年",
    "2013年",
    "2014年",
    "2015年",
    "2016年",
    "2017年",
    "2018年",
    "2019年",
    "2020年",
    "2021年",
    "2022年",
    "2023年",
    "2024年",
]


class GeoMap:
    def __init__(self, file_path: str):
        """
        初始化地图对象

        Args:
            file_path (str): 地图路径
        """
        self.gdf: GeoDataFrame = gpd.read_file(file_path).to_crs(
            epsg=4326
        )  # 提取地图文件为 GeoDataFrame 格式，将gdf坐标系转换为经纬度坐标系
        self.geo_info = self.gdf.geometry  # 提取地图文件中的地理几何对象信息
        self.outline = self.geo_info.iloc[0]  # 假设第一个几何对象就是地图边界
        self.lon = (self.outline.bounds[0] + self.outline.bounds[2]) / 2
        self.lat = (self.outline.bounds[1] + self.outline.bounds[3]) / 2
        self.dots_df: pd.DataFrame | None = None
        self.dots_list: List[Dot] = []
        self.params_name: List[str] = []
        self.grid = None  # 初始默认未进行栅格化

    def is_inside(self, lon: float, lat: float) -> bool:
        """
        判断点位是否在地图范围内

        Args:
            lon (float): 经度
            lat (float): 纬度

        Returns:
            bool: 是否在地图范围内
        """
        point = Point(lon, lat)
        is_within = point.within(self.outline)
        return is_within

    def grid_paint(self, grid_row: int, grid_col: int):
        """
        将地图划分为 grid_row * grid_col 的栅格

        Args:
            grid_row (int): 栅格行数
            grid_col (int): 栅格列数

        Returns:
            Grid: 返回的栅格对象
        """

        # 获取边界范围
        minx, miny, maxx, maxy = self.outline.bounds
        width = maxx - minx  # 经度范围
        height = maxy - miny  # 纬度范围

        cell_width = width / grid_row
        cell_height = height / grid_col
        self.grid = Grid(minx, miny, cell_width, cell_height, grid_row, grid_col, self)

        return self.grid

    def load_dots_df(self, dots_df: pd.DataFrame, params_name: List[str] = params_name):
        """
        从 DataFrame 中加载点位数据

        Args:
            dots_df (pd.DataFrame): 包含点位数据的 DataFrame
            params_name (List[str], optional): DataFrame 的列名. Defaults to params_name.

        Raises:
            ValueError: dots_df 必须包含 '东经' 和 '北纬' 列
        """
        self.params_name = params_name
        # 检查输入是否包含必要列
        if space[0] not in dots_df.columns or space[1] not in dots_df.columns:
            raise ValueError("dots_df 必须包含 '东经' 和 '北纬' 列")

        if self.dots_df is None:
            self.dots_df = dots_df
        else:
            self.dots_df = pd.concat([self.dots_df, dots_df], ignore_index=True)

        for _, line in dots_df.iterrows():
            # line is pd.Series
            lon, lat = line[space[0]], line[space[1]]
            try:
                dot = Dot(
                    lon=lon,
                    lat=lat,
                    bg_map=self,
                    bg_grid=self.grid,
                    dot_type=1,
                    params=line[self.params_name].to_dict(),
                )

                self.dots_list.append(dot)
                if self.grid:
                    self.grid.load_dots(dot)
            except ValueError as e:
                print(e)

        if self.grid:
            self.grid.set_focus()

    def save_risk_image(self, label: str, speed: float = 0, year: int = 0) -> str:
        """
        保存风险区域图像

        Args:
            speed (float): 增长速率
            year (int): 预测年份

        Raises:
            ValueError: 栅格没有初始化

        Returns:
            str: 返回的图片文件路径
        """
        if self.grid is None:
            raise ValueError("栅格没有初始化,请调用 grid_paint() 初始化栅格.")
        self.grid.set_value_matrix(label)
        bool_matrix = self.grid.value_matrix + speed * year > 0.2

        fig, ax = plt.subplots()
        self.gdf.boundary.plot(ax=ax, color="black", linewidth=1)

        bounds = self.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        # 自定义颜色映射，True 区域为红色，False 区域为透明
        cmap = ListedColormap(["none", "red"])

        img = ax.imshow(bool_matrix, extent=bounds, origin="lower", alpha=0.5, cmap=cmap)

        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        new_year = int(label.replace("年", "")) + year
        ax.set_title(f"风险区域图({new_year}年)")

        file_path = os.path.join(get_temp_image_path(), "img_risk_area.png")
        fig.savefig(file_path)
        plt.close()

        return file_path

    def save_grid_image(self, label: str = "Fe"):
        """
        保存指定元素的栅格图像

        Args:
            label (str): 要绘制栅格图的元素标签. Defaults to "Fe".

        Returns:
            str: 返回的图片文件路径
        """
        if self.grid is None:
            raise ValueError("栅格没有初始化,请调用 grid_paint() 初始化栅格.")
        self.grid.set_value_matrix(label)
        fig, ax = plt.subplots()
        self.gdf.boundary.plot(ax=ax, color="black", linewidth=1)

        bounds = self.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        # 获取值的最小值和最大值
        vmin = np.nanmin(self.grid.value_matrix)
        vmax = np.nanmax(self.grid.value_matrix)

        # 计算80%部分的值
        vcenter = np.nanpercentile(self.grid.value_matrix, 80)

        # 使用 TwoSlopeNorm 进行颜色归一化
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        img = ax.imshow(self.grid.value_matrix, extent=bounds, origin="lower", alpha=0.5, norm=norm)

        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.colorbar(img, ax=ax)

        file_path = os.path.join(get_temp_image_path(), f"img_grid_{label}.png")

        fig.savefig(file_path)
        plt.close()

        return file_path

    def save_grid_image_simple(self, label: str = "Fe", color: bool = True):
        """
        保存指定元素的栅格图像，不含有坐标轴

        Args:
            label (str, optional): 显示的元素标签. Defaults to "Fe".
            color (bool, optional): 是彩图还是灰度图. Defaults to True.

        Raises:
            ValueError: 没有进行栅格初始化

        Returns:
            str: 返回的图片文件路径
        """

        if self.grid is None:
            raise ValueError("栅格没有初始化,请调用 grid_paint() 初始化栅格.")
        self.grid.set_value_matrix(label)
        fig, ax = plt.subplots()

        cmap = "viridis" if color else "Greys"
        alpha = 0.5 if color else 1

        norm = None

        if color:
            # 获取值的最小值和最大值
            vmin = np.nanmin(self.grid.value_matrix)
            vmax = np.nanmax(self.grid.value_matrix)

            # 计算80%部分的值
            vcenter = np.nanpercentile(self.grid.value_matrix, 80)

            # 使用 TwoSlopeNorm 进行颜色归一化
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        ax.imshow(self.grid.value_matrix, origin="lower", cmap=cmap, alpha=alpha, norm=norm)
        ax.axis("off")

        tag = "grey" if not color else "color"
        file_path = os.path.join(get_temp_image_path(), f"img_grid_{label}_{tag}.png")
        fig.savefig(file_path)
        plt.close()

        return file_path

    def save_dot_image(self, label_basic: str = "2008年", label: str = "2012年"):
        """
        保存点位地图

        Args:
            label_basic (str, optional): 选择基准年份. Defaults to "2008年".
            label (str, optional): 选择保存的点位年份. Defaults to "2012年".

        Raises:
            ValueError: 没有加载点位数据

        Returns:
            str: 返回的图片文件路径
        """

        if self.dots_df is None:
            raise ValueError("没有加载点位数据")

        # 计算两个标签的比值
        df = self.dots_df
        df["ratio"] = df[label] / df[label_basic]

        fig, ax = plt.subplots()

        scatter = plt.scatter(df["东经"], df["北纬"], c=df["ratio"], cmap="viridis", alpha=0.6)
        fig.colorbar(scatter, label=f"增加百分比值", ax=ax)

        bounds = self.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        self.gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel("东经")
        ax.set_ylabel("北纬")

        file_path = os.path.join(get_temp_image_path(), f"img_dot_{label_basic}_to_{label}.png")

        fig.savefig(file_path)
        plt.close()

        return file_path


class Grid:
    def __init__(self, init_x: float, init_y: float, size_x: float, size_y: float, row_num: int, col_num: int, bg: GeoMap):
        """
        初始化栅格对象

        Args:
            init_x (float): 栅格初始化点的经度
            init_y (float): 栅格初始化点的纬度
            size_x (float): 栅格的宽度
            size_y (float): 栅格的高度
            row_num (int): 栅格的行数
            col_num (int): 栅格的列数
            bg (GeoMap): 栅格归属的背景地图
        """
        self.init_point = Point(init_x, init_y)  # 定义栅格的初始点，一般为左上方点位
        self.init_x = init_x
        self.init_y = init_y
        self.size_x = size_x
        self.size_y = size_y  # 小栅格的宽度、高度
        self.row_num = row_num
        self.col_num = col_num  # 栅格的行数、列数
        self.bg: GeoMap = bg  # 归属的背景地图
        self.cell_matrix: List[List[Optional[Cell]]] = [[None] * col_num for _ in range(row_num)]
        self.value_matrix: np.ndarray[float] = np.zeros((row_num, col_num), dtype=float)
        self.__init_cell()  #  赋值 cell_matrix & 中心点 & 有效性
        self.__syn_dots_list()
        self.set_focus()

    def __repr__(self):
        return (
            f"Grid(init_point:{self.init_point}, size_x:{self.size_x}, "
            f"size_y:{self.size_y}, row_num:{self.row_num}, col_num:{self.col_num})"
        )

    def get_cell_pos(self, x: float, y: float) -> tuple[int, int]:
        """
        获取点位所在的栅格位置

        Args:
            x (float): 点位的经度
            y (float): 点位的纬度

        Raises:
            ValueError: 点位超出栅格范围

        Returns:
            tuple[int, int]: 返回栅格的行列位置
        """
        # 获取栅格的行列
        if (self.init_x <= x <= (self.init_x + self.col_num * self.size_x)) and (
            self.init_y <= y <= (self.init_y + self.row_num * self.size_y)
        ):
            col = floor((x - self.init_x) / self.size_x)
            row = floor((y - self.init_y) / self.size_y)
        else:
            raise ValueError(f"点位({x},{y})超出栅格范围")
        return row, col

    def __init_cell(self):
        """
        初始化栅格中的每个小栅格
        """
        # 遍历每行和每列，计算每个小栅格的中心点
        for row in range(self.row_num):
            for col in range(self.col_num):
                # 中心点的坐标
                center_x = self.init_x + (col + 0.5) * self.size_x
                center_y = self.init_y + (row + 0.5) * self.size_y

                is_valid = self.bg.is_inside(center_x, center_y)

                self.cell_matrix[row][col] = Cell(
                    lon=center_x,
                    lat=center_y,
                    bg_map=self.bg,
                    bg_grid=self,
                    is_valid=is_valid,
                )

    def __syn_dots_list(self):
        """
        将点位数据同步到栅格中
        """
        for dot in self.bg.dots_list:
            dot.bg_grid = self
            self.load_dots(dot)

    def load_dots(self, dot):
        """
        将点位数据加载到栅格中

        Args:
            dot (Dot): 点位数据
        """
        row, col = self.get_cell_pos(dot.lon, dot.lat)
        self.cell_matrix[row][col].dots_list.append(dot)

    def set_focus(self):
        """
        计算栅格中每个小栅格的平均值点位
        """
        for row in self.cell_matrix:
            for cell in row:
                cell.focus_dot = cell.calc_focus()

    def set_value_matrix(self, label: str):
        """
        将栅格中的点位参数值同步到栅格的数值矩阵中

        Args:
            label (str): 参数标签

        Raises:
            ValueError: 需要选择合法的标签

        Returns:
            NDArry: 返回的数值矩阵
        """
        if label not in self.bg.params_name:
            raise ValueError("需要选择合法的标签")
        for row in range(self.row_num):
            for col in range(self.col_num):
                self.value_matrix[row][col] = (
                    self.cell_matrix[row][col].focus_dot.params[label]
                    if self.cell_matrix[row][col].focus_dot and self.cell_matrix[row][col].is_valid
                    else np.nan
                )
        return self.value_matrix


class Cell:
    def __init__(self, lon: float, lat: float, bg_map: GeoMap, bg_grid: Grid, is_valid: bool = True):
        """
        初始化栅格中的小栅格

        Args:
            lon (float): 小栅格的经度
            lat (float): 小栅格的纬度
            bg_map (GeoMap): 小栅格的背景地图
            bg_grid (Grid): 小栅格的背景栅格
            is_valid (bool, optional): 小栅格是否在地图边界内. Defaults to True.
        """
        self.lon: float = lon
        self.lat: float = lat
        self.dots_list: List[Dot] = []
        self.focus_dot: Optional[Dot] = None
        self.is_valid: bool = is_valid
        self.bg_map: GeoMap = bg_map
        self.bg_grid: Grid = bg_grid

    def __repr__(self):
        return f"Cell(lon={self.lon}, lat={self.lat}, is_valid={self.is_valid})"

    def calc_focus(self):
        """
        计算栅格中的点位的平均值

        Returns:
            Dot: 平均值点位
        """
        if self.dots_list:
            avg_lon = sum(dot.lon for dot in self.dots_list) / len(self.dots_list)
            avg_lat = sum(dot.lat for dot in self.dots_list) / len(self.dots_list)
            average_params = self.__get_average_params()

            return Dot(lon=avg_lon, lat=avg_lat, bg_map=self.bg_map, bg_grid=self.bg_grid, dot_type=2, params=average_params)

    def __get_average_params(self):
        """
        计算点位的平均参数

        Returns:
            dict: 平均参数字典
        """
        params_list = [dot.params for dot in self.dots_list if dot.params]

        all_keys = set(params_list[0].keys())

        average_dict = {key: 0 for key in all_keys}

        for params in params_list:
            for key, value in params.items():
                average_dict[key] += value

        num = len(params_list)
        for key in average_dict:
            average_dict[key] /= num

        return average_dict


class Dot:
    def __init__(self, lon: float, lat: float, bg_map: GeoMap, bg_grid: Grid, dot_type: int, params: Optional[dict] = None):
        """
        初始化点位对象

        Args:
            lon (float): 点位的经度
            lat (float): 点位的纬度
            bg_map (GeoMap): 点位的背景地图
            bg_grid (Grid): 点位的背景栅格
            dot_type (int): 点位类型
            params (Optional[dict], optional): 点位的参数. Defaults to None.
        """
        self.lon = lon  # 点位经度
        self.lat = lat  # 点位纬度
        self.bg_map = bg_map  # 点位归属背景地图
        self.bg_grid = bg_grid  # 点位归属栅格
        # 以下属性暂未用到
        self.dot_type = dot_type  # type为1表示普通散点，type为2表示focus
        self.params = params  # 节点参数字典

    def __repr__(self):
        return f"Dot(lon={self.lon}, lat={self.lat}, params={self.params})"
