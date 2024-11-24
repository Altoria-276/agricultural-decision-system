import os
from typing import List, Optional
import matplotlib.pyplot as plt
from numpy._typing._array_like import NDArray
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from alphashape import alphashape
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


def kringing(df: pd.DataFrame, num):
    outpath = os.path.join(".", "kringing", "data")
    lon, lat = df[space[0]], df[space[1]]

    min_lon, max_lon = lon.min(), lon.max()
    min_lat, max_lat = lat.min(), lat.max()

    lon, lat = df[space[0]], df[space[1]]

    # 创建一个空GeoDataFrame用于存储插值结果
    geometry = [
        Point(xy) for xy in zip(np.tile(np.linspace(min_lon, max_lon, num), num), np.repeat(np.linspace(min_lat, max_lat, num), num))
    ]
    crs = {"init": "epsg:4326"}  # EPSG:4326坐标系
    interpolated_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs=crs, geometry=geometry)

    # 循环遍历每个属性进行插值
    for param in params_name:
        data = df[param]
        OK = OrdinaryKriging(lon, lat, data, variogram_model="spherical", nlags=3)
        z, ss = OK.execute("grid", np.linspace(min_lon, max_lon, num), np.linspace(min_lat, max_lat, num))
        interpolated_gdf[param] = z.flatten()

    # 保存插值结果为shapefile文件和数据表
    interpolated_gdf.to_file(f"{outpath}.shp")

    # 提取经度和纬度信息
    interpolated_gdf[space[0]] = interpolated_gdf.geometry.apply(lambda geom: geom.x)
    interpolated_gdf[space[1]] = interpolated_gdf.geometry.apply(lambda geom: geom.y)

    # 将经度和纬度列放在前两列
    interpolated_gdf = interpolated_gdf[space + params_name]
    interpolated_gdf.to_excel(f"{outpath}.xlsx", index=False)
    print("插值数据保存成功！")
    return f"{outpath}.xlsx"


class Map:
    def __init__(self, file_path):
        self.gdf: GeoDataFrame = gpd.read_file(file_path).to_crs(
            epsg=4326
        )  # 提取地图文件为 GeoDataFrame 格式，将gdf坐标系转换为经纬度坐标系
        self.geo_info = self.gdf.geometry  # 提取地图文件中的地理几何对象信息
        self.outline = self.geo_info.iloc[0]  # 假设第一个几何对象就是地图边界
        # self.outline = self.gdf.boundary.to_crs(epsg=4326)     # 地图边界第二种写法
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

    def save_image(self, label: str = "K"):
        fig, ax = plt.subplots()
        self.gdf.plot(ax=ax, facecolor="none", edgecolor="black")
        self.grid.set_value_matrix(label)

        bounds = map.outline.bounds
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        img = ax.imshow(map.grid.value_matrix, extent=bounds, origin="lower", alpha=0.5)

        map.gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.colorbar(img, ax=ax)
        fig.savefig("./Images/img.png")


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

    def load_dots(self, dots_df: pd.DataFrame):
        """
        将散点映射到栅格中，散点之后会默认进行 focus 计算
        :param dots_df: 包含点位经纬度的列 'lon' 和 'lat'。
        :return: None
        """
        # 检查输入是否包含必要列
        if space[0] not in dots_df.columns or space[1] not in dots_df.columns:
            raise ValueError("dots_df 必须包含 '东经' 和 '北纬' 列")

        # 遍历点位，映射到栅格
        for _, line in dots_df.iterrows():
            # line is pd.Series
            lon, lat = line[space[0]], line[space[1]]
            try:
                row, col = self.get_cell_pos(lon, lat)

                dot = Dot(
                    lon=lon,
                    lat=lat,
                    bg_map=self.bg,
                    bg_grid=self,
                    dot_type=1,
                    params=line[params_name].to_dict(),
                )
                self.cell_matrix[row][col].dots_list.append(dot)
            except ValueError as e:
                print(e)

    def set_focus(self):
        # set focus
        for row in self.cell_matrix:
            for cell in row:
                cell.focus_dot = cell.calc_focus()

    def set_value_matrix(self, label: str):
        if label not in params_name:
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


def run():
    file_path_shp = os.path.join(".", "数据", "湘潭县界.shp")

    kringing(pd.read_excel(os.path.join(".", "数据", "水稻点位148.xlsx")), 100)

    file_path_df = os.path.join(".", "数据", "水稻点位148.xlsx")
    file_path_kringing = os.path.join(".", "kringing", "data.xlsx")

    map = Map(file_path_shp)
    map.grid_paint(grid_row=30, grid_col=30)
    grid = map.grid
    grid.load_dots(pd.read_excel(file_path_kringing))
    grid.load_dots(pd.read_excel(file_path_df))
    grid.set_focus()

    fig, ax = plt.subplots()
    map.gdf.plot(ax=ax, facecolor="none", edgecolor="black")
    grid.set_value_matrix("Cd")

    bounds = map.outline.bounds
    bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

    img = ax.imshow(map.grid.value_matrix, extent=bounds, origin="lower", alpha=0.5)

    map.gdf.boundary.plot(ax=ax, color="black", linewidth=1)
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(img, ax=ax)

    fig.savefig("./Images/img.png")


if __name__ == "__main__":
    run()

# """克里金插值"""

#
# def kringing(self, df, attributes, longitude, latitude, file_name, outpath, num):
#     # 读取指定范围的shapefile文件
#     # specified_shapefile = gpd.read_file(maskBoundary)
#
#     # 获取经度和纬度的最大最小值
#     # min_longitude = specified_shapefile.bounds['minx'].min()
#     # max_longitude = specified_shapefile.bounds['maxx'].max()
#     # min_latitude = specified_shapefile.bounds['miny'].min()
#     # max_latitude = specified_shapefile.bounds['maxy'].max()
#
#     min_longitude, max_longitude = df[space[0]].min(), df[space[0]].max()
#     min_latitude, max_latitude = df[space[1]].min(), df[space[1]].max()
#
#     # 创建一个空GeoDataFrame用于存储插值结果
#     geometry = [
#         Point(xy)
#         for xy in zip(
#             np.tile(np.linspace(min_longitude, max_longitude, num), num), np.repeat(np.linspace(min_latitude, max_latitude, num), num)
#         )
#     ]
#     crs = {"init": "epsg:4326"}  # EPSG:4326坐标系
#     interpolated_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs=crs, geometry=geometry)
#
#     # 循环遍历每个属性进行插值
#     for attribute in attributes:
#         data = df[attribute]
#         OK = OrdinaryKriging(longitude, latitude, data, variogram_model="spherical", nlags=3)
#         z, ss = OK.execute("grid", np.linspace(min_longitude, max_longitude, num), np.linspace(min_latitude, max_latitude, num))
#         interpolated_gdf[attribute] = z.flatten()
#
#     # 保存插值结果为shapefile文件和数据表
#     interpolated_data_shp_path = f"./{outpath}/{file_name}.shp"
#     interpolated_gdf.to_file(interpolated_data_shp_path)
#
#     # 提取经度和纬度信息
#     interpolated_gdf["经度"] = interpolated_gdf.geometry.apply(lambda geom: geom.x)
#     interpolated_gdf["纬度"] = interpolated_gdf.geometry.apply(lambda geom: geom.y)
#
#     # 将经度和纬度列放在前两列
#     interpolated_gdf = interpolated_gdf[["经度", "纬度"] + attributes]
#
#     interpolated_data_xlsx_path = f"./{outpath}/{file_name}.xlsx"
#     interpolated_gdf.to_excel(interpolated_data_xlsx_path, index=False, engine="xlsxwriter")
#     print("插值数据保存成功！")
#     print("文件名称")
#     print(f"./{outpath}/{file_name}.xlsx")
#     return f"./{outpath}", f"./{outpath}/{file_name}.xlsx"
#
#
# """散点与SHP映射"""
#
#
# def poi2image(self, all_file, cluster_folder, excel_name, outfile_path):
#     alldata = pd.read_excel(all_file)
#
#     # 创建绘图对象
#     fig, ax = plt.subplots()
#     # 绘制shp边界
#     data.plot(ax=ax, facecolor="none", edgecolor="black")
#     m = 1
#     c = 1
#     handles = []
#     labels = []
#     for j in excel_name:
#         df = pd.read_excel(f"./{cluster_folder}/{j}")
#
#         # 经度和纬度数据
#         longitude = df.loc[:, "经度"]
#         latitude = df.loc[:, "纬度"]
#
#         # 将散点数据转换为二维坐标数组
#         points = np.column_stack((longitude, latitude))
#         # 使用Alpha Shapes算法计算边界
#         alpha = 50  # Alpha参数，用于控制边界的平滑程度,越大生成的区域越多
#         boundary = alphashape(points, alpha)
#
#         # 绘制边界
#         if isinstance(boundary, Polygon):
#             expanded_polygon = boundary.buffer(0.003)
#             # 单个多边形边界
#             x, y = expanded_polygon.exterior.xy
#             # boundary_x = [x for x, y in boundary.exterior.coords]
#             # boundary_y = [y for x, y in boundary.exterior.coords]
#             # 使用B-spline曲线拟合
#             # tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
#             # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
#             # u_new = np.linspace(u.min(), u.max(), 1000)
#             # x_new, y_new = interpolate.splev(u_new, tck)
#             # 绘制平滑后的曲线
#             # plt.plot(x_new, y_new, color=colors[j])
#             ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
#             handles.append(ax.scatter([], [], color=colors[c]))
#             # labels.append(j)
#             gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])
#             gdf.crs = "EPSG:4326"
#
#             # 设置输出文件的路径和名称
#             output_file_name = "boundary" + str(m) + ".shp"
#             outfilepath = f"./{outfile_path}/{output_file_name}"
#             m = m + 1
#
#             # 将GeoDataFrame保存为shp文件
#             gdf.to_file(outfilepath, driver="ESRI Shapefile")
#             df = pd.DataFrame()
#
#             for a in alldata.index:
#                 point = Point(alldata.loc[a, "经度"], alldata.loc[a, "纬度"])
#                 if point.within(boundary):
#                     df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
#                     df = pd.concat([df, df1], axis=0, ignore_index=True)
#             if len(df) > 20:
#                 excel_name_out = "区域" + str(m - 1) + ".xlsx"
#                 labels_name = "区域" + str(m - 1)
#                 labels.append(labels_name)
#                 excel_path_out = f"./{outfile_path}/{excel_name_out}"
#                 df.to_excel(excel_path_out, index=False)
#             else:
#                 m = m - 1
#
#         elif isinstance(boundary, MultiPolygon):
#             # 多个多边形边界
#             # 绘制多边形
#             for polygon in boundary.geoms:
#                 # boundary_x = [x for x, y in polygon.exterior.coords]
#                 # boundary_y = [y for x, y in polygon.exterior.coords]
#                 # 使用B-spline曲线拟合
#                 # tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
#                 # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
#                 # u_new = np.linspace(u.min(), u.max(), 1000)
#                 # x_new, y_new = interpolate.splev(u_new, tck)
#                 # 绘制平滑后的曲线
#                 # plt.plot(x_new, y_new, color=colors[j])
#                 expanded_polygon = polygon.buffer(0.003)
#                 x, y = expanded_polygon.exterior.xy
#                 ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
#                 handles.append(ax.scatter([], [], color=colors[c]))
#                 # labels.append(j)
#                 # print(expanded_polygon)
#                 gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])
#
#                 gdf.crs = "EPSG:4326"
#                 # 设置输出文件的路径和名称
#                 output_file_name = "boundary" + str(m) + ".shp"
#                 outfilepath = f"./{outfile_path}/{output_file_name}"
#                 m = m + 1
#                 # 将GeoDataFrame保存为shp文件
#                 gdf.to_file(outfilepath, driver="ESRI Shapefile")
#
#                 df = pd.DataFrame()
#                 for a in alldata.index:
#                     point = Point(alldata.loc[a, "经度"], alldata.loc[a, "纬度"])
#                     if point.within(polygon):
#                         df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
#                         df = pd.concat([df, df1], axis=0, ignore_index=True)
#
#                 if len(df) > 20:
#                     excel_name_out = "区域" + str(m - 1) + ".xlsx"
#                     labels_name = "区域" + str(m - 1)
#                     labels.append(labels_name)
#                     excel_path_out = f"./{outfile_path}/{excel_name_out}"
#                     df.to_excel(excel_path_out, index=False)
#                 else:
#                     m = m - 1
#
#         # 绘制边界
#         # plt.plot(*boundary.exterior.xy, color=colors[j])
#         # 绘制散点图
#         plt.scatter(longitude, latitude, color=colors[c], s=1)
#         c = c + 1
#     # 绘制多边形
#     # data.plot(ax=ax, facecolor='none', edgecolor='black')
#     data.plot(ax=ax, facecolor="none", edgecolor="black")
#     handles = list(set(handles))
#     labels = list(set(labels))
#     plt.legend(handles, labels, loc="upper left")
#
#     # 设置坐标轴范围
#     ax.set_xlim(data.total_bounds[0] - 0.2, data.total_bounds[2] + 0.01)
#     ax.set_ylim(data.total_bounds[1] - 0.01, data.total_bounds[3] + 0.01)
#
#     plt.axis("off")  # 不显示坐标轴
#     # 设置图形属性
#     # ax.set_aspect('equal')  # 保持纵横比相等
#
#     plt.savefig(f"./final_result/过程{m-1}.png")
#     # 显示图形
#     # plt.show()
#     return m - 1
#
#
# """栅格化"""
#
#
# def grid_paint(kriexcel_list, grid_size, image_name, shp_file):
#
#     # 创建图形和轴对象
#     fig, ax = plt.subplots()
#
#     i = 1
#     for excelname, shapefile in zip(kriexcel_list, shp_file):
#
#         # 读取shp文件
#         shapefile = gpd.read_file(shapefile)
#
#         # 获取shp文件的几何形状，这里假设只有一个多边形
#         polygon = shapefile.geometry.values[0]
#
#         df = pd.read_excel(excelname)  # 读取Excel数据
#
#         # 确定地理范围和栅格大小
#         lon_min, lon_max = df[space[0]].min(), df[space[0]].max()
#         lat_min, lat_max = df[space[1]].min(), df[space[1]].max()
#
#         # 计算栅格的行数和列数
#         num_lon_bins = int(np.ceil((lon_max - lon_min) / grid_size))
#         num_lat_bins = int(np.ceil((lat_max - lat_min) / grid_size))
#
#         # 创建一个二维数组来表示栅格，以及一个用于计数的数组
#         grid_sum = np.zeros((num_lat_bins, num_lon_bins))
#         grid_count = np.zeros((num_lat_bins, num_lon_bins))
#
#         # 将数据点分配到栅格并计算值的总和和计数
#         for _, row in df.iterrows():
#             lon = row[space[0]]
#             lat = row[space[1]]
#             value = row[label]
#             lon_bin = int((lon - lon_min) / grid_size)
#             lat_bin = int((lat - lat_min) / grid_size)
#             grid_sum[lat_bin, lon_bin] += value
#             grid_count[lat_bin, lon_bin] += 1
#
#             # 计算平均值
#         grid = np.ma.masked_where(grid_count == 0, grid_sum / grid_count)
#
#         # 创建一个mask数组，根据shp文件的形状将多边形内部的栅格设为True，外部设为False
#         mask = np.zeros((num_lat_bins, num_lon_bins), dtype=bool)
#         for i in range(num_lat_bins):
#             for j in range(num_lon_bins):
#                 # 将栅格的坐标转换为经纬度
#                 lon = lon_min + j * grid_size
#                 lat = lat_min + i * grid_size
#                 # 检查这个经纬度点是否在多边形内部
#                 point = Point(lon, lat)
#                 if polygon.contains(point):
#                     mask[i, j] = True
#
#                     # 使用mask数组遮挡栅格图上的特定区域
#         grid = np.ma.masked_where(~mask, grid)  # 使用~mask来反转mask数组，遮挡多边形外部的区域
#
#         i = i + 1
#         # 绘制栅格图，并获取返回的图像对象
#         img = ax.imshow(grid, extent=[lon_min, lon_max, lat_min, lat_max], origin="lower", alpha=0.5)
#
#         # 绘制Shapefile的边界
#         shapefile.boundary.plot(ax=ax, color="red", linewidth=1)
#         # 显示图形并保存图像文件
#         # 添加颜色条
#
#     # 绘制地图边界
#     data = gpd.read_file(AmapPath)
#     data.boundary.plot(ax=ax, color="black", linewidth=1)
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.colorbar(img, ax=ax)
#     # plt.show()
#     save_path = "./final_result/" + image_name + ".png"
#     plt.savefig(save_path)
