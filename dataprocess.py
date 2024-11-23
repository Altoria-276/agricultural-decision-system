import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from alphashape import alphashape
from shapely.geometry import Polygon, MultiPolygon, Point


np.random.seed(42)  # 设置种子值为42
plt.rcParams["font.sans-serif"] = ["SimHei"]    # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False      # 用来正常显示负号

warnings.filterwarnings("ignore")
# 全局，错误数据的索引
globals_dict = {}
globals_right = {}
# 显示所有列
pd.set_option("display.max_columns", None)
# 显示所有行
pd.set_option("display.max_rows", None)

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略UserWarning

# 颜色列表
colors = [
    "red",
    "cyan",
    "pink",
    "orange",
    "limegreen",
    "salmon",
    "grey",
    "gold",
    "darkgreen",
    "royalblue",
    "darkmagenta",
    "darkgoldenrod",
    "maroon",
    "saddlebrown",
    "lawngreen",
    "olive",
    "navy",
]

space = ["东经", "北纬"]
label = []          # label为重金属
col = []            # col为环境协变量
kringing_num = 80   # 插值50x50
logpre_excel = []   # 克里金插值后生成的excel名称列表，预测,重金属做差，之后保存成一个新的excel名称列表
grid_size = 0.01    # 设置栅格大小
rmse_thres = 0.4    # rmse_thres为阈值
# shapefile = './数据/湘潭镇界.shp'
# data = gpd.read_file(shapefile)
endframe = pd.DataFrame()


class Map:
    def __init__(self, file_path):
        self.gdf = gpd.read_file(file_path).to_crs(epsg=4326)   # 提取地图文件为GeoDataFrame格式，将gdf坐标系转换为经纬度坐标系
        self.geo_info = self.gdf.geometry                       # 提取地图文件中的地理几何对象信息
        self.outline = self.geo_info.iloc[0]                    # 假设第一个几何对象就是地图边界
        # self.outline = self.gdf.boundary.to_crs(epsg=4326)     # 地图边界第二种写法
        self.grid = None                                        # 初始默认未进行栅格化

    def is_inside(self, lon, lat):
        """
        用于判断给定经纬度的点位是否位于此地图边界内
        :param lon: 给定点位的经度
        :param lat: 给定点位的纬度
        :return: 返回bool型变量，表示是否位于边界内
        """
        point = Point(lon, lat)
        is_within = point.within(self.outline)
        return is_within

    def grid_paint(self, grid_row, grid_col):
        """

        :param grid_row: 需要绘制的栅格行数
        :param grid_col: 需要绘制的栅格列数
        :return : None
        """

        # 获取边界范围
        minx, miny, maxx, maxy = self.outline.bounds
        width = maxx - minx     # 经度范围
        height = maxy - miny    # 纬度范围

        cell_width = width / grid_row
        cell_height = height / grid_col
        self.grid = Grid(minx, maxy, cell_width, cell_height, grid_row, grid_col, self)


class Grid:
    def __init__(self, init_x, init_y, size_x, size_y, row_num, col_num, bg):
        self.init_point = Point(init_x, init_y)         # 定义栅格的初始点，一般为左上方点位
        self.init_x = init_x
        self.init_y = init_y
        self.size_x = size_x
        self.size_y = size_y        # 小栅格的宽度、高度
        self.row_num = row_num
        self.col_num = col_num      # 栅格的行数、列数
        self.bg = bg                # 归属的背景地图
        self.center_matrix = self.get_cell_center()         # 每个小栅格的中心点矩阵
        self.valid_matrix = self.get_valid_matrix(bg)       # 有效性矩阵（表示每个小栅格是否位于地图内）
        # 以下属性在散点之前默认初始为空
        self.dots_list = []
        self.focus_matrix = [[]]

    def __repr__(self):
        return (f"Grid(init_point:{self.init_point}, size_x:{self.size_x}, "
                f"size_y:{self.size_y}, row_num:{self.row_num}, col_num:{self.col_num})")

    def get_cell_index(self, row, col):
        if (row < self.row_num) and (col < self.col_num):
            cell_index = (row - 1) * self.col_num + col
        else:
            raise IndexError("索引超出栅格范围")
        return cell_index

    def get_cell_pos(self, x, y):
        """
        判断给定坐标点位位于栅格中的哪一小格
        :param x: 给定点位经度
        :param y: 给定点位维度
        :return: 点位位置小格的索引编号, TODO:注意，小栅格的编号从1开始，不能为0或负数
        """
        if (self.init_x <= x <= (self.init_x + self.col_num * self.size_x)) and ((self.init_y - self.row_num * self.size_y) <= y <= self.init_y):
            col = int((x - self.init_x) / self.size_x) + 1
            row = int((self.init_y - y) / self.size_y) + 1
        else:
            raise ValueError("点位超出栅格范围")
        return self.get_cell_index(row, col)

    def get_cell_center(self):
        # 初始化一个空的二维矩阵
        center_matrix = []

        # 遍历每行和每列，计算每个小栅格的中心点
        for row in range(self.row_num):
            row_centers = []
            for col in range(self.col_num):
                # 中心点的坐标
                center_x = self.init_x + (col + 0.5) * self.size_x
                center_y = self.init_y - (row + 0.5) * self.size_y
                row_centers.append((center_x, center_y))  # 存储为元组
            center_matrix.append(row_centers)

        return center_matrix

    def get_valid_matrix(self, bg):
        """
        遍历中心点矩阵，生成判断小栅格有效性的布尔矩阵
        :param bg: 背景地图，一个Map类的实例化对象
        :return: 布尔矩阵（二维列表），表示每个小栅格的有效性
        """
        valid_matrix = []

        # 遍历中心点矩阵，判断每个点是否在范围内
        for row in self.center_matrix:
            valid_row = []
            for center_x, center_y in row:
                valid_row.append(bg.is_inside(center_x, center_y))
            valid_matrix.append(valid_row)

        return valid_matrix

    def dots_mapping(self, dots_df):
        """
        将散点映射到栅格中，散点之后会默认进行focus矩阵的计算
        :param dots_df: 包含点位经纬度的列 'lon' 和 'lat'。
        :return: None
        """
        # 检查输入是否包含必要列
        if space[0] not in dots_df.columns or space[1] not in dots_df.columns:
            raise ValueError("dots_df 必须包含 '东经' 和 '北纬' 列")

        # 清空 dots_list
        self.dots_list = []

        i = 0
        # 遍历点位，映射到栅格
        for _, row in dots_df.iterrows():
            lon, lat = row[space[0]], row[space[1]]
            try:
                cell_index = self.get_cell_pos(lon, lat)
                dot = Dot(lon=lon, lat=lat, cell_index=cell_index, bg_map=self.bg, bg_grid=self, index=i, dot_type=1)
                self.dots_list.append(dot)
            except ValueError as e:
                print(f"点位 ({lon}, {lat}) 超出栅格范围，跳过")
            i += 1

        self.focus_matrix = self.get_focus_matrix()

    def reverse_mapping(self, cell_index):
        """
        根据栅格编号反向映射，获取该栅格中的散点列表
        :param cell_index: 小栅格编号
        :return: 该栅格中的散点列表，若无散点则返回空列表
        """
        if cell_index < 1 or cell_index > self.row_num * self.col_num:
            raise ValueError("栅格编号超出范围")

        # 查找 dots_list 中所属栅格为 cell_index 的点位
        points_in_cell = [dot for dot in self.dots_list if dot.cell_index == cell_index]

        return points_in_cell

    def calc_focus(self, cell_index):
        """
        计算指定栅格的focus点位
        :param cell_index: 栅格编号
        :return: Dot 对象，表示栅格的 focus；若栅格内无点，则返回 None
        """
        points_in_cell = self.reverse_mapping(cell_index)
        if not points_in_cell:  # 如果没有点
            return None
        else:  # 如果有多个点，计算平均值
            avg_lon = sum(dot.lon for dot in points_in_cell) / len(points_in_cell)
            avg_lat = sum(dot.lat for dot in points_in_cell) / len(points_in_cell)
            return Dot(avg_lon, avg_lat, cell_index, bg_map=self.bg, bg_grid=self, index=cell_index, dot_type=2)

    def get_focus_matrix(self):
        """
        生成 focus 矩阵，每个元素为 Dot 对象或 None。
        :return: 二维列表，表示每个栅格的 focus
        """
        focus_matrix = []
        for row in range(1, self.row_num + 1):
            focus_row = []
            for col in range(1, self.col_num + 1):
                cell_index = self.get_cell_index(row, col)
                focus = self.calc_focus(cell_index)
                focus_row.append(focus)
            focus_matrix.append(focus_row)
        return focus_matrix

    def interpolation(self):
        """
        将栅格中的focus虚拟点进行插值
        """
        for focus_row in self.focus_matrix:
            for focus_dot in focus_row:
                if focus_dot:
                    self.dots_list.append(focus_dot)


class Dot:
    def __init__(self, lon, lat, cell_index, bg_map, bg_grid, index, dot_type):
        self.lon = lon  # 点位经度
        self.lat = lat  # 点位纬度
        self.cell_index = cell_index    # 点位归属的栅格编号
        self.bg_map = bg_map            # 点位归属背景地图
        self.bg_grid = bg_grid          # 点位归属栅格
        # 以下属性暂未用到
        self.index = index              # 点位编号
        self.dot_type = dot_type        # type为1表示普通散点，type为2表示focus

    def __repr__(self):
        return f"Dot(lon={self.lon}, lat={self.lat}, cell_index={self.cell_index}, index={self.index})"




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
