from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from math import ceil, floor
from geopandas.geodataframe import GeoDataFrame

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
    "基准",
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


class Map:
    def __init__(self, file_path):
        self.gdf: GeoDataFrame = gpd.read_file(file_path).to_crs(
            epsg=4326
        )  # 提取地图文件为 GeoDataFrame 格式，将gdf坐标系转换为经纬度坐标系
        self.geo_info = self.gdf.geometry  # 提取地图文件中的地理几何对象信息
        self.outline = self.geo_info.iloc[0]  # 假设第一个几何对象就是地图边界
        # self.outline = self.gdf.boundary.to_crs(epsg=4326)     # 地图边界第二种写法
        self.dots_list: List[Dot] = []
        self.params_name: List[str] = []
        self.grid = None  # 初始默认未进行栅格化

    def is_inside(self, lon: float, lat: float) -> bool:
        """
        用于判断给定经纬度的点位是否位于此地图边界内
        :param lon: 给定点位的经度
        :param lat: 给定点位的纬度
        :return: 返回bool型变量，表示是否位于边界内
        """
        point = Point(lon, lat)
        is_within = point.within(self.outline)
        return is_within

    def grid_paint(self, grid_row: int, grid_col: int):
        """

        :param grid_row: 需要绘制的栅格行数
        :param grid_col: 需要绘制的栅格列数
        :return : None
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
        self.params_name = params_name
        # 检查输入是否包含必要列
        if space[0] not in dots_df.columns or space[1] not in dots_df.columns:
            raise ValueError("dots_df 必须包含 '东经' 和 '北纬' 列")

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

    def save_grid_image(self, label: str = "K"):
        fig, ax = plt.subplots()
        self.gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        # self.gdf.plot(ax=ax, facecolor="none", edgecolor="black")
        self.grid.set_value_matrix(label)

        bounds = self.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        img = ax.imshow(map.grid.value_matrix, extent=bounds, origin="lower", alpha=0.5)

        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.colorbar(img, ax=ax)
        fig.savefig("./Images/img2.png")

    def save_dot_image(self, label: str = "2012年"):
        fig, ax = plt.subplots()
        # self.gdf.plot(ax=ax, facecolor="none", edgecolor="black")
        for dot in self.dots_list:
            ax.scatter(dot.lon, dot.lat, c="violet")

        bounds = self.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        self.gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.savefig("./Images/img3.png")


class Grid:
    def __init__(self, init_x: float, init_y: float, size_x: float, size_y: float, row_num: int, col_num: int, bg: Map):
        self.init_point = Point(init_x, init_y)  # 定义栅格的初始点，一般为左上方点位
        self.init_x = init_x
        self.init_y = init_y
        self.size_x = size_x
        self.size_y = size_y  # 小栅格的宽度、高度
        self.row_num = row_num
        self.col_num = col_num  # 栅格的行数、列数
        self.bg: Map = bg  # 归属的背景地图
        self.cell_matrix: List[List[Optional[Cell]]] = [[None] * col_num for _ in range(row_num)]
        self.value_matrix: np.ndarray[float] = np.zeros((row_num, col_num), dtype=float)
        self.__init_cell()  #  赋值 cell_matrix & 中心点 & 有效性
        self.__syn_dots_list()

    def __repr__(self):
        return (
            f"Grid(init_point:{self.init_point}, size_x:{self.size_x}, "
            f"size_y:{self.size_y}, row_num:{self.row_num}, col_num:{self.col_num})"
        )

    def get_cell_pos(self, x: float, y: float):
        # 获取栅格的行列
        if (self.init_x <= x <= (self.init_x + self.col_num * self.size_x)) and (
            self.init_y <= y <= (self.init_y + self.row_num * self.size_y)
        ):
            col = floor((x - self.init_x) / self.size_x)
            row = floor((y - self.init_y) / self.size_y)
        else:
            raise ValueError("点位超出栅格范围")
        return row, col

    def __init_cell(self):
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
        for dot in self.bg.dots_list:
            dot.bg_grid = self
            self.load_dots(dot)

    def load_dots(self, dot):
        row, col = self.get_cell_pos(dot.lon, dot.lat)
        self.cell_matrix[row][col].dots_list.append(dot)

    def set_focus(self):
        # set focus
        for row in self.cell_matrix:
            for cell in row:
                cell.focus_dot = cell.calc_focus()

    def set_value_matrix(self, label: str):
        if label not in self.bg.params_name:
            raise ValueError("需要选择合法的标签")
        for row in range(self.row_num):
            for col in range(self.col_num):
                self.value_matrix[row][col] = (
                    self.cell_matrix[row][col].focus_dot.params[label]
                    if self.cell_matrix[row][col].focus_dot and self.cell_matrix[row][col].is_valid
                    else np.nan
                )

    def conv_interpolation(self):
        update = True
        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while update:
            update = False
            for i in range(1, self.row_num - 1):
                for j in range(1, self.col_num - 1):
                    if not self.cell_matrix[i][j].focus_dot:
                        neighbors = [(i + di, j + dj) for di, dj in neighbors_offsets]
                        neighbors_dots: List[Optional[Dot]] = []

                        for row, col in neighbors:
                            if self.cell_matrix[row][col].focus_dot:
                                neighbors_dots.append(self.cell_matrix[row][col].focus_dot)

                        if neighbors_dots:
                            dot = self.cell_matrix[i][j].calc_focus_interpolation(neighbors_dots)
                            if self.get_cell_pos(dot.lon, dot.lat) == (row, col):
                                self.cell_matrix[i][j].focus_dot = dot
                                update = True


class Cell:
    def __init__(self, lon: float, lat: float, bg_map: Map, bg_grid: Grid, is_valid: bool = True):
        self.lon: float = lon
        self.lat: float = lat
        self.dots_list: List[Dot] = []
        self.focus_dot: Optional[Dot] = None
        self.is_valid: bool = is_valid
        self.bg_map: Map = bg_map
        self.bg_grid: Grid = bg_grid

    def __repr__(self):
        return f"Cell(lon={self.lon}, lat={self.lat}, is_valid={self.is_valid})"

    def calc_focus_interpolation(self, dots_list):
        self.dots_list = dots_list
        dot = self.calc_focus()
        dot.dot_type = 3
        self.dots_list = []
        return dot

    def calc_focus(self):
        if self.dots_list:
            avg_lon = sum(dot.lon for dot in self.dots_list) / len(self.dots_list)
            avg_lat = sum(dot.lat for dot in self.dots_list) / len(self.dots_list)
            average_params = self.__get_average_params()

            return Dot(lon=avg_lon, lat=avg_lat, bg_map=self.bg_map, bg_grid=self.bg_grid, dot_type=2, params=average_params)

    def __get_average_params(self):
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
    def __init__(self, lon: float, lat: float, bg_map: Map, bg_grid: Grid, dot_type: int, params: Optional[dict] = None):
        self.lon = lon  # 点位经度
        self.lat = lat  # 点位纬度
        self.bg_map = bg_map  # 点位归属背景地图
        self.bg_grid = bg_grid  # 点位归属栅格
        # 以下属性暂未用到
        self.dot_type = dot_type  # type为1表示普通散点，type为2表示focus
        self.params = params  # 节点参数字典

    def __repr__(self):
        return f"Dot(lon={self.lon}, lat={self.lat}, params={self.params})"
