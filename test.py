import shutil
from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtWidgets import QFileDialog
from PySide2.QtUiTools import QUiLoader
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from alphashape import alphashape
from shapely.geometry import Polygon, MultiPolygon, Point
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from PySide2.QtWidgets import QMainWindow
from PySide2.QtGui import QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QComboBox, QListWidget, QCheckBox, QListWidgetItem, QLineEdit, QApplication
from PySide2.QtGui import Qt
from PySide2.QtCore import Signal
import sys
import re

np.random.seed(42)  # 设置种子值为42
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False
# 用来正常显示负号

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

space = ["经度", "纬度"]
label = []  ##label为重金属
col = []  ##col为环境协变量
method = "KMeans"
kringing_num = 80  # 插值50x50
logpre_excel = []  # 克里金插值后生成的excel名称列表，预测,重金属做差，之后保存成一个新的excel名称列表
grid_size = 0.01  # 设置栅格大小
rmse_thres = 0.4  # rmse_thres为阈值
# shapefile = './数据/湘潭镇界.shp'
# data = gpd.read_file(shapefile)
endframe = pd.DataFrame()


class allp:
    # 生成带有时间戳的文件夹
    def create_timestamped_folder(self, prefix):
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # 创建以时间戳命名的文件夹
        folder_name = f"process_documents/{prefix}_{timestamp}"
        os.makedirs(folder_name)

        return folder_name

    def create_result_folder(self):
        folder_path = "./final_result"
        # 检查文件夹是否已存在
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
        else:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    # 删除文件
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    # 删除子文件夹
                    shutil.rmtree(item_path)

    def get_excel_filenames(self, directory):
        xls_files = glob.glob(os.path.join(directory, "*.xlsx"))
        filenames = [os.path.basename(file) for file in xls_files]
        return filenames

    def get_shp_filenames(self, directory):
        shp_files = glob.glob(os.path.join(directory, "*.shp"))
        filenames = [os.path.basename(file) for file in shp_files]
        return filenames

    def logistic_regression_test(self, all_data, label, x_col):
        y_col = []
        y_col.append(label)
        Y = all_data.loc[:, y_col]
        X = all_data.loc[:, x_col]
        col_all = x_col.copy()
        np.concatenate([col_all, y_col], axis=0)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        # 计算均方误差（MSE）
        mse = mean_squared_error(y_test, y_predict)
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_predict)
        print("正在回归:", label)
        print("算法rmse:", rmse)

        return rmse, mse, r2

    def logi_exam(self, excel_file):
        x_col = col
        y_col = label
        all_data = pd.read_excel(excel_file)
        # 归一化
        columns_to_normalize = x_col + y_col
        scaler = MinMaxScaler()
        all_data[columns_to_normalize] = scaler.fit_transform(all_data[columns_to_normalize])
        flag = 0
        for i in y_col:
            r, m, r2 = self.logistic_regression_test(all_data, i, x_col)
            if r >= rmse_thres:
                flag = 1
        if flag == 0:
            print("{}通过检验".format(excel_file))
            return
        if flag == 1:
            print("{}未通过检验".format(excel_file))
            return excel_file

    """克里金插值"""

    def kringing(self, df, attributes, longitude, latitude, file_name, outpath, num):
        # 读取指定范围的shapefile文件
        # specified_shapefile = gpd.read_file(maskBoundary)

        # 获取经度和纬度的最大最小值
        # min_longitude = specified_shapefile.bounds['minx'].min()
        # max_longitude = specified_shapefile.bounds['maxx'].max()
        # min_latitude = specified_shapefile.bounds['miny'].min()
        # max_latitude = specified_shapefile.bounds['maxy'].max()

        min_longitude, max_longitude = df[space[0]].min(), df[space[0]].max()
        min_latitude, max_latitude = df[space[1]].min(), df[space[1]].max()

        # 创建一个空GeoDataFrame用于存储插值结果
        geometry = [
            Point(xy)
            for xy in zip(
                np.tile(np.linspace(min_longitude, max_longitude, num), num), np.repeat(np.linspace(min_latitude, max_latitude, num), num)
            )
        ]
        crs = {"init": "epsg:4326"}  # EPSG:4326坐标系
        interpolated_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs=crs, geometry=geometry)

        # 循环遍历每个属性进行插值
        for attribute in attributes:
            data = df[attribute]
            OK = OrdinaryKriging(longitude, latitude, data, variogram_model="spherical", nlags=3)
            z, ss = OK.execute("grid", np.linspace(min_longitude, max_longitude, num), np.linspace(min_latitude, max_latitude, num))
            interpolated_gdf[attribute] = z.flatten()

        # 保存插值结果为shapefile文件和数据表
        interpolated_data_shp_path = f"./{outpath}/{file_name}.shp"
        interpolated_gdf.to_file(interpolated_data_shp_path)

        # 提取经度和纬度信息
        interpolated_gdf["经度"] = interpolated_gdf.geometry.apply(lambda geom: geom.x)
        interpolated_gdf["纬度"] = interpolated_gdf.geometry.apply(lambda geom: geom.y)

        # 将经度和纬度列放在前两列
        interpolated_gdf = interpolated_gdf[["经度", "纬度"] + attributes]

        interpolated_data_xlsx_path = f"./{outpath}/{file_name}.xlsx"
        interpolated_gdf.to_excel(interpolated_data_xlsx_path, index=False, engine="xlsxwriter")
        print("插值数据保存成功！")
        print("文件名称")
        print(f"./{outpath}/{file_name}.xlsx")
        return f"./{outpath}", f"./{outpath}/{file_name}.xlsx"

    def cluster_with_methods(self, covariate_data, clu_folder, n_cluster=None):
        # 读取协变量数据为 Pandas DataFrame
        df = pd.read_excel(covariate_data)

        # 选择要用于聚类的属性列
        # selected_columns = ['Pb', 'Cd', 'As_', 'Cr', 'Hg', 'Zn', 'Cu', 'K', 'P', 'N', 'Si', 'Se', 'Na', 'Mo', 'Mn', 'Mg', 'La',
        #                    'Fe', 'Ca', 'Al', 'Ni', 'Ti', 'pH', 'CEC', 'SOM']

        selected_columns = label + col
        # 提取选定的属性列数据
        data = df[selected_columns].values

        # 执行不同的聚类算法
        if method == "KMeans":
            if n_cluster is None:
                raise ValueError("参数 n_cluster 不能为空！")
            clustering_method = KMeans(n_clusters=n_cluster)
            labels = clustering_method.fit_predict(data)
        elif method == "Agglomerative":
            if n_cluster is None:
                raise ValueError("参数 n_cluster 不能为空！")
            clustering_method = AgglomerativeClustering(n_clusters=n_cluster)
            labels = clustering_method.fit_predict(data)
        elif method == "Spectral":
            if n_cluster is None:
                raise ValueError("参数 n_cluster 不能为空！")
            clustering_method = SpectralClustering(n_clusters=n_cluster, affinity="nearest_neighbors", n_neighbors=10)
            labels = clustering_method.fit_predict(data)
        elif method == "GMM":
            if n_cluster is None:
                raise ValueError("参数 n_cluster 不能为空！")
            clustering_method = GaussianMixture(n_components=n_cluster)
            labels = clustering_method.fit_predict(data)
        else:
            raise ValueError("不支持的聚类方法！")

        # 将聚类结果添加到 DataFrame
        df["Cluster"] = labels

        # 根据聚类结果拆分 DataFrame
        dfs = []
        for cluster_id in range(len(set(labels))):
            cluster_df = df[df["Cluster"] == cluster_id].copy()
            dfs.append(cluster_df)

        # 保存每个类别的 DataFrame 到独立的 Excel 文件
        for i, cluster_df in enumerate(dfs):
            output_file = f"./{clu_folder}/cluster_{i+1}.xlsx"
            cluster_df.to_excel(output_file, index=False)

    def mul_cluster(self, covariate_data, clu_folder, n_cluster):
        # 读取协变量数据为 Pandas DataFrame
        df = pd.read_excel(covariate_data)

        # 选择要用于聚类的属性列
        # selected_columns = ['Pb', 'Cd', 'As_', 'Cr', 'Hg', 'Zn', 'Cu', 'K', 'P', 'N', 'Si', 'Se', 'Na', 'Mo', 'Mn', 'Mg', 'La',
        #                   'Fe', 'Ca', 'Al', 'Ni', 'Ti', 'pH', 'CEC', 'SOM']
        selected_columns = col + label
        # 提取选定的属性列数据
        data = df[selected_columns].values

        # 执行 K-means 聚类算法
        num_clusters = n_cluster  # 聚类数量
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        labels = kmeans.labels_

        # 将聚类结果添加到 DataFrame
        df["Cluster"] = labels

        # 根据聚类结果拆分 DataFrame
        dfs = []
        for cluster_id in range(num_clusters):
            cluster_df = df[df["Cluster"] == cluster_id].copy()
            dfs.append(cluster_df)

        # 保存每个类别的 DataFrame 到独立的 Excel 文件
        for i, cluster_df in enumerate(dfs):
            output_file = f"./{clu_folder}/cluster_{i+1}.xlsx"
            cluster_df.to_excel(output_file, index=False)

    # 散点与SHP映射
    def poi2image(self, all_file, cluster_folder, excel_name, outfile_path):
        alldata = pd.read_excel(all_file)

        # 创建绘图对象
        fig, ax = plt.subplots()
        # 绘制shp边界
        data.plot(ax=ax, facecolor="none", edgecolor="black")
        m = 1
        c = 1
        handles = []
        labels = []
        for j in excel_name:
            df = pd.read_excel(f"./{cluster_folder}/{j}")

            # 经度和纬度数据
            longitude = df.loc[:, "经度"]
            latitude = df.loc[:, "纬度"]

            # 将散点数据转换为二维坐标数组
            points = np.column_stack((longitude, latitude))
            # 使用Alpha Shapes算法计算边界
            alpha = 50  # Alpha参数，用于控制边界的平滑程度,越大生成的区域越多
            boundary = alphashape(points, alpha)

            # 绘制边界
            if isinstance(boundary, Polygon):
                expanded_polygon = boundary.buffer(0.003)
                # 单个多边形边界
                x, y = expanded_polygon.exterior.xy
                # boundary_x = [x for x, y in boundary.exterior.coords]
                # boundary_y = [y for x, y in boundary.exterior.coords]
                # 使用B-spline曲线拟合
                # tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
                # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
                # u_new = np.linspace(u.min(), u.max(), 1000)
                # x_new, y_new = interpolate.splev(u_new, tck)
                # 绘制平滑后的曲线
                # plt.plot(x_new, y_new, color=colors[j])
                ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
                handles.append(ax.scatter([], [], color=colors[c]))
                # labels.append(j)
                gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])
                gdf.crs = "EPSG:4326"

                # 设置输出文件的路径和名称
                output_file_name = "boundary" + str(m) + ".shp"
                outfilepath = f"./{outfile_path}/{output_file_name}"
                m = m + 1

                # 将GeoDataFrame保存为shp文件
                gdf.to_file(outfilepath, driver="ESRI Shapefile")
                df = pd.DataFrame()

                for a in alldata.index:
                    point = Point(alldata.loc[a, "经度"], alldata.loc[a, "纬度"])
                    if point.within(boundary):
                        df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
                        df = pd.concat([df, df1], axis=0, ignore_index=True)
                if len(df) > 20:
                    excel_name_out = "区域" + str(m - 1) + ".xlsx"
                    labels_name = "区域" + str(m - 1)
                    labels.append(labels_name)
                    excel_path_out = f"./{outfile_path}/{excel_name_out}"
                    df.to_excel(excel_path_out, index=False)
                else:
                    m = m - 1

            elif isinstance(boundary, MultiPolygon):
                # 多个多边形边界
                # 绘制多边形
                for polygon in boundary.geoms:
                    # boundary_x = [x for x, y in polygon.exterior.coords]
                    # boundary_y = [y for x, y in polygon.exterior.coords]
                    # 使用B-spline曲线拟合
                    # tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
                    # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
                    # u_new = np.linspace(u.min(), u.max(), 1000)
                    # x_new, y_new = interpolate.splev(u_new, tck)
                    # 绘制平滑后的曲线
                    # plt.plot(x_new, y_new, color=colors[j])
                    expanded_polygon = polygon.buffer(0.003)
                    x, y = expanded_polygon.exterior.xy
                    ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
                    handles.append(ax.scatter([], [], color=colors[c]))
                    # labels.append(j)
                    # print(expanded_polygon)
                    gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])

                    gdf.crs = "EPSG:4326"
                    # 设置输出文件的路径和名称
                    output_file_name = "boundary" + str(m) + ".shp"
                    outfilepath = f"./{outfile_path}/{output_file_name}"
                    m = m + 1
                    # 将GeoDataFrame保存为shp文件
                    gdf.to_file(outfilepath, driver="ESRI Shapefile")

                    df = pd.DataFrame()
                    for a in alldata.index:
                        point = Point(alldata.loc[a, "经度"], alldata.loc[a, "纬度"])
                        if point.within(polygon):           # TODO--判断点是否在多边形内
                            df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
                            df = pd.concat([df, df1], axis=0, ignore_index=True)

                    if len(df) > 20:
                        excel_name_out = "区域" + str(m - 1) + ".xlsx"
                        labels_name = "区域" + str(m - 1)
                        labels.append(labels_name)
                        excel_path_out = f"./{outfile_path}/{excel_name_out}"
                        df.to_excel(excel_path_out, index=False)
                    else:
                        m = m - 1

            # 绘制边界
            # plt.plot(*boundary.exterior.xy, color=colors[j])
            # 绘制散点图
            plt.scatter(longitude, latitude, color=colors[c], s=1)
            c = c + 1
        # 绘制多边形
        # data.plot(ax=ax, facecolor='none', edgecolor='black')
        data.plot(ax=ax, facecolor="none", edgecolor="black")
        handles = list(set(handles))
        labels = list(set(labels))
        plt.legend(handles, labels, loc="upper left")

        # 设置坐标轴范围
        ax.set_xlim(data.total_bounds[0] - 0.2, data.total_bounds[2] + 0.01)
        ax.set_ylim(data.total_bounds[1] - 0.01, data.total_bounds[3] + 0.01)

        plt.axis("off")  # 不显示坐标轴
        # 设置图形属性
        # ax.set_aspect('equal')  # 保持纵横比相等

        plt.savefig(f"./final_result/过程{m-1}.png")
        # 显示图形
        # plt.show()
        return m - 1

    def log_test(self, n, excel_name):
        all_excel = []
        fail_excel = []
        for i in range(1, n + 1):
            excel_file = excel_name + str(i) + ".xlsx"
            all_excel.append(excel_file)
        for exc in all_excel:
            fail = self.logi_exam(exc)
            if fail:
                fail_excel.append(fail)
        suce_excel = [x for x in all_excel if x not in fail_excel]
        print("通过检验的excel为:")
        print(suce_excel)
        print("未通过检验的excel为:")
        print(fail_excel)
        return suce_excel, fail_excel

    def log_test_2(self, rmse_thres, excel_path, excel_list):
        all_excel = []
        fail_excel = []
        for i in excel_list:
            excel_file = f"./{excel_path}/{i}"
            all_excel.append(excel_file)
        for exc in all_excel:
            fail = self.logi_exam(exc, rmse_thres)
            if fail:
                fail_excel.append(fail)
        suce_excel = [x for x in all_excel if x not in fail_excel]
        print("通过检验的excel为:")
        print(suce_excel)
        print("未通过检验的excel为:")
        print(fail_excel)
        return suce_excel, fail_excel

    def check_values_within_range(self, excel_file1, excel_file2):  # 第一个为插值excel
        # 读取两个Excel文件为DataFrame
        df1 = pd.read_excel(excel_file1)
        df2 = pd.read_excel(excel_file2)

        # 用于存储不符合范围的行
        rows_outside_range = []

        # 遍历第一个Excel中的每一行
        for index, row in df1.iterrows():
            # 遍历第二个Excel中的每一列
            for column in df2.columns:
                # 获取第二个Excel中当前列的均值和标准差
                mean = df2[column].mean()
                std = df2[column].std()

                # 检查第一个Excel中当前行的值是否在均值加减3倍标准差的范围内
                value = row[column]
                if value < mean - 3 * std or value > mean + 3 * std:
                    rows_outside_range.append((index, column, value, mean, std))

        # 输出不符合范围的行
        if len(rows_outside_range) > 0:
            print("以下行的值不在范围内：")
            for row_info in rows_outside_range:
                index, column, value, mean, std = row_info
                print(f"行 {index + 1} 列 {column}：值 {value}，均值 {mean}，标准差 {std}")
        else:
            print("所有值都在范围内")
        row_ind = []
        for i in range(len(rows_outside_range)):
            row_ind.append(rows_outside_range[i][0])
        row_index = list(set(row_ind))
        print(list(set(row_ind)))

        return rows_outside_range, row_index  # 第一个为详细信息，包括没通过的行数，列名 #第二个只没通过的行数

    def result_processing(self, excel_name):
        fig, ax = plt.subplots()

        data.plot(ax=ax, facecolor="none", edgecolor="black")

        c = 1
        handles = []
        labels = []
        for j in excel_name:
            df = pd.read_excel(j)

            longitude = df.loc[:, "经度"]
            latitude = df.loc[:, "纬度"]

            points = np.column_stack((longitude, latitude))
            try:
                alpha = 50
                boundary = alphashape(points, alpha)

                if isinstance(boundary, Polygon):
                    expanded_polygon = boundary.buffer(0.003)
                    x, y = expanded_polygon.exterior.xy
                    ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
                    handles.append(ax.scatter([], [], color=colors[c]))
                    parts = j.split("/")
                    region = parts[-1].split(".")[0]  # 去掉扩展名部分
                    labels.append(region)

                elif isinstance(boundary, MultiPolygon):
                    for polygon in boundary.geoms:
                        expanded_polygon = polygon.buffer(0.003)
                        x, y = expanded_polygon.exterior.xy
                        ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
                        handles.append(ax.scatter([], [], color=colors[c]))
                        parts = j.split("/")
                        region = parts[-1].split(".")[0]  # 去掉扩展名部分
                        labels.append(region)
                # 绘制散点图
                # plt.scatter(longitude, latitude, color=colors[c], s=1)
                c = c + 1

            except:
                print("{}区域过小".format(j))

        data.plot(ax=ax, facecolor="none", edgecolor="black")

        # 设置坐标轴范围
        ax.set_xlim(data.total_bounds[0] - 0.2, data.total_bounds[2] + 0.01)
        ax.set_ylim(data.total_bounds[1] - 0.01, data.total_bounds[3] + 0.01)

        plt.axis("off")
        handles = list(set(handles))
        labels = list(set(labels))
        plt.legend(handles, labels, loc="upper left")  # 将图例定位到右上角

        plt.savefig("./final_result/result.png")
        # plt.show()


def data_processing(path, label, col):
    # 使用 pd.ExcelFile 打开 Excel 文件

    xls = pd.ExcelFile(path)

    # 读取 Excel 文件中的第一个表格
    df_y = pd.read_excel(xls, xls.sheet_names[0])
    df_x = pd.read_excel(xls, xls.sheet_names[1])
    data_x = df_x[col]
    data_y = df_y[space + label]
    df = pd.concat([data_y, data_x], axis=1)
    # # 检查 'As' 列是否存在
    # if 'As' in df.columns:
    #     # 替换 'As' 列的值
    #     df = df.rename(columns={'As': 'As_'})

    # # 检查 'Cd有效态' 列是否存在
    # if 'Cd有效态' in df.columns:
    #     # 替换 'Cd1' 列的值
    #     df = df.rename(columns={'Cd有效态': 'Cd1'})
    #
    # # 检查 'Cd水稻' 列是否存在
    # if 'Cd水稻' in df.columns:
    #     # 替换 'Cd1' 列的值
    #     df = df.rename(columns={'Cd水稻': 'Cd2'})

    # 获取原始文件的目录路径
    dir_path = os.path.dirname(path)

    # 拼接新文件的保存路径
    new_path = os.path.join(dir_path, "test.xlsx")

    # 保存为新的 Excel 文件
    # df = df[field]

    df.to_excel(new_path, index=False)
    # 返回新文件的路径和所在地址
    return new_path


# def grid_paint(kriexcel_list,grid_size,image_name):
#     # 创建图形和轴对象
#     fig, ax = plt.subplots()
#
#     for excelname in kriexcel_list:
#         df = pd.read_excel(excelname) # 读取Excel数据
#
#         # 确定地理范围和栅格大小
#         lon_min, lon_max = df[space[0]].min(), df[space[0]].max()
#         lat_min, lat_max = df[space[1]].min(), df[space[1]].max()
#
#         # 计算栅格的行数和列数
#         num_lon_bins = int(np.ceil((lon_max - lon_min) / grid_size))
#         num_lat_bins = int(np.ceil((lat_max - lat_min) / grid_size))
#
#         # 创建一个二维数组来表示栅格
#         grid = np.zeros((num_lat_bins, num_lon_bins))
#
#         # 将数据点分配到栅格并计算值的总和
#         for _, row in df.iterrows():
#             lon = row[space[0]]
#             lat = row[space[1]]
#             value = row[label]
#             lon_bin = int((lon - lon_min) / grid_size)
#             lat_bin = int((lat - lat_min) / grid_size)
#             grid[lat_bin, lon_bin] += value
#
#         # 绘制栅格图
#         plt.imshow(grid, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower')
#
#     # 绘制地图边界
#     data = gpd.read_file(AmapPath)
#     data.plot(ax=ax, facecolor='none', edgecolor='black')
#
#     plt.colorbar(label=label)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Grid Map')
#
#     save_path = './final_result/'+image_name+'.png'
#     plt.savefig(save_path)
#     #plt.show() # 显示图形

"""栅格化"""


def grid_paint(kriexcel_list, grid_size, image_name, shp_file):

    # 创建图形和轴对象
    fig, ax = plt.subplots()

    i = 1
    for excelname, shapefile in zip(kriexcel_list, shp_file):

        # 读取shp文件
        shapefile = gpd.read_file(shapefile)

        # 获取shp文件的几何形状，这里假设只有一个多边形
        polygon = shapefile.geometry.values[0]

        df = pd.read_excel(excelname)  # 读取Excel数据

        # 确定地理范围和栅格大小
        lon_min, lon_max = df[space[0]].min(), df[space[0]].max()
        lat_min, lat_max = df[space[1]].min(), df[space[1]].max()

        # 计算栅格的行数和列数
        num_lon_bins = int(np.ceil((lon_max - lon_min) / grid_size))
        num_lat_bins = int(np.ceil((lat_max - lat_min) / grid_size))

        # 创建一个二维数组来表示栅格，以及一个用于计数的数组
        grid_sum = np.zeros((num_lat_bins, num_lon_bins))
        grid_count = np.zeros((num_lat_bins, num_lon_bins))

        # 将数据点分配到栅格并计算值的总和和计数
        for _, row in df.iterrows():
            lon = row[space[0]]
            lat = row[space[1]]
            value = row[label]
            lon_bin = int((lon - lon_min) / grid_size)
            lat_bin = int((lat - lat_min) / grid_size)
            grid_sum[lat_bin, lon_bin] += value
            grid_count[lat_bin, lon_bin] += 1

            # 计算平均值
        grid = np.ma.masked_where(grid_count == 0, grid_sum / grid_count)

        # 创建一个mask数组，根据shp文件的形状将多边形内部的栅格设为True，外部设为False
        mask = np.zeros((num_lat_bins, num_lon_bins), dtype=bool)
        for i in range(num_lat_bins):
            for j in range(num_lon_bins):
                # 将栅格的坐标转换为经纬度
                lon = lon_min + j * grid_size
                lat = lat_min + i * grid_size
                # 检查这个经纬度点是否在多边形内部
                point = Point(lon, lat)
                if polygon.contains(point):
                    mask[i, j] = True

                # 使用mask数组遮挡栅格图上的特定区域
        grid = np.ma.masked_where(~mask, grid)  # 使用~mask来反转mask数组，遮挡多边形外部的区域

        i = i + 1
        # 绘制栅格图，并获取返回的图像对象
        img = ax.imshow(grid, extent=[lon_min, lon_max, lat_min, lat_max], origin="lower", alpha=0.5)

        # 绘制Shapefile的边界
        shapefile.boundary.plot(ax=ax, color="red", linewidth=1)
        # 显示图形并保存图像文件
        # 添加颜色条

    # 绘制地图边界
    data = gpd.read_file(AmapPath)
    data.boundary.plot(ax=ax, color="black", linewidth=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(img, ax=ax)
    # plt.show()
    save_path = "./final_result/" + image_name + ".png"
    plt.savefig(save_path)


def logistic_pre(kriexcel_name):

    all_data = pd.read_excel(kriexcel_name)

    Y = all_data.loc[:, label]
    X = all_data.loc[:, col]
    col_all = col.copy()
    np.concatenate([col_all, col], axis=0)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    # 将y_test和y_predict转换为相同的数据类型
    y_test = np.array(y_test, dtype=np.float64)
    y_predict = np.array(y_predict, dtype=np.float64)

    # 计算均方误差（MSE）
    mse = mean_squared_error(y_test, y_predict)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)

    # 计算每个数据点的绝对误差
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(y_test, y_predict)]
    # 计算平均绝对误差
    mae = np.mean(absolute_errors)
    # 打印MAE值

    parts = kriexcel_name.split("/")
    region = parts[-1].split(".")[0]  # 去掉扩展名部分
    print(region)

    print("正在回归:", region)
    print("算法rmse:", rmse)
    print("MAE:", mae)
    print("r2:", r2)

    # 删除列
    all_data = all_data.drop(label, axis=1)
    y_label = model.predict(X)
    y_label = y_label - Y  # 误差
    all_data[label] = y_label
    newexcel_name = kriexcel_name.replace(".xlsx", "") + "_dif" + ".xlsx"
    all_data.to_excel(newexcel_name, index=False)
    logpre_excel.append(newexcel_name)

    region = region.replace("区域", "")
    region = "region" + region
    endframe.index = ["rmse", "mae", "r2"]
    endframe[region] = [rmse, mae, r2]

    return rmse, mae, r2


def run():
    global data
    print(method)
    data = gpd.read_file(AmapPath)
    ap = allp()
    field = space + label + col
    all_success_file = []
    all_fail_file = []

    stats.ui.progressBar.setValue(0)  # 进度条

    # 创建结果文件夹
    ap.create_result_folder()
    ################聚类################
    stats.ui.progressBar.setValue(1)  # 进度条
    data_excel = AfilePath  # 总excel
    data_excel = data_processing(data_excel, label, col)

    cluster_folder = ap.create_timestamped_folder("cluster")
    num_cluster = 5
    ap.cluster_with_methods(data_excel, cluster_folder, num_cluster)
    stats.ui.progressBar.setValue(2)  # 进度条
    ################区划################
    excelName = ap.get_excel_filenames(cluster_folder)
    boundary_folder = ap.create_timestamped_folder("division")
    n = ap.poi2image(data_excel, cluster_folder, excelName, boundary_folder)

    ###############回归检测##############
    stats.ui.progressBar.setValue(3)  # 进度条
    # rmse_thres = 0.4
    excel_name = f"./{boundary_folder}/区域"
    suce_excel, fail_excel = ap.log_test(n, excel_name)
    all_success_file = all_success_file + suce_excel
    all_fail_file = all_fail_file + fail_excel

    suce_excel2 = []
    fail_excel2 = []

    cluster_folder_fail = ap.create_timestamped_folder("cluster_again")
    boundary_folder_fail = ap.create_timestamped_folder("division_again")
    kringing_ex_folder = ap.create_timestamped_folder("kringing")

    kriexcel_list = []
    print("重金属：", label)

    ############未通过区域再聚类###########
    for item in fail_excel:
        i_label = 0
        num_cluster_ = 3
        ap.cluster_with_methods(item, cluster_folder_fail, num_cluster_)

        ##############未通过区域做区划###############
        excelName_ = ap.get_excel_filenames(cluster_folder_fail)
        n_ = ap.poi2image(data_excel, cluster_folder_fail, excelName_, boundary_folder_fail)

        ##############插值#############
        # 依次读取未通过区域划分的excel文件进行打点，shp文件进行掩膜
        eName = ap.get_excel_filenames(boundary_folder_fail)
        print(eName)
        sName = ap.get_shp_filenames(boundary_folder_fail)
        print(sName)

        for a in eName:
            try:
                # 读取表格数据
                path = f"./{boundary_folder_fail}/{a}"
                df = pd.read_excel(path)

                # 属性列表
                attributes = [field_ for field_ in field if field_ not in ["经度", "纬度"]]

                print("属性:", attributes)

                longitude = df["经度"]
                latitude = df["纬度"]
                a = a[:-5]
                k_path, exk = ap.kringing(df, attributes, longitude, latitude, a, kringing_ex_folder, kringing_num)
                print(exk)
                kriexcel_list.append(exk)

                ############插值检验###########
                all_row, row = ap.check_values_within_range(exk, f"./{boundary_folder_fail}/{a}")
                # count = 0
                # while(len(row) != 0):
                #     for f in field:
                #         kringing()
                #         tif2shp()
                #     shp2excel()
                #     all_row, row = check_values_within_range()
                #     count = count + 1
                #     if len(row) == 0 or count == 3:
                #         continue
                ###############插值结果回归检测##############
                rmse_thres = 0.4
                suce_excel2, fail_excel2 = ap.log_test_2(rmse_thres, kringing_ex_folder, [a])
            except:
                print("./{}/{}出错".format(boundary_folder_fail, sName[i_label]))

            if i_label < len(sName):
                i_label = i_label + 1

    print(kriexcel_list)
    stats.ui.progressBar.setValue(4)  # 进度条
    print(all_success_file)
    ap.result_processing(all_success_file + suce_excel2)
    print("first time clustering success: {}".format(all_success_file))
    print("first time clustering fail: {}".format(all_fail_file))
    print("second time clustering success: {}".format(suce_excel2))
    print("second time clustering fail: {}".format(fail_excel2))

    print(boundary_folder)
    sName = ap.get_shp_filenames(boundary_folder)
    print(sName)
    shp_file = []  # shp_file为完整边界存储路径
    for i in sName:
        shpname = boundary_folder + "/" + i
        shp_file.append(shpname)
    print(shp_file)

    i_label = 0

    for a in all_success_file:
        print(a)
        try:
            # 读取表格数据
            df = pd.read_excel(a)

            parts = a.split("/")
            region = parts[-1].split(".")[0]  # 去掉扩展名部分
            print(region)

            # 属性列表
            attributes = [field_ for field_ in field if field_ not in ["经度", "纬度"]]
            longitude = df["经度"]
            latitude = df["纬度"]

            k_path, exk = ap.kringing(df, attributes, longitude, latitude, region, kringing_ex_folder, kringing_num)

            print(exk)
            kriexcel_list.append(exk)

            ############插值检验###########
            all_row, row = ap.check_values_within_range(exk, f"./{boundary_folder_fail}/{region}")
            # count = 0
            # while(len(row) != 0):
            #     for f in field:
            #         kringing()
            #         tif2shp()
            #     shp2excel()
            #     all_row, row = check_values_within_range()
            #     count = count + 1
            #     if len(row) == 0 or count == 3:
            #         continue
            ###############插值结果回归检测##############
            rmse_thres = 0.4
            suce_excel2, fail_excel2 = ap.log_test_2(rmse_thres, kringing_ex_folder, [region])
        except:
            print("./{}/{}出错".format(boundary_folder_fail, sName[i_label]))

        if i_label < len(sName):
            i_label = i_label + 1

    for i in kriexcel_list:
        logistic_pre(i)

    grid_paint(kriexcel_list, grid_size, "gridmap", shp_file)
    grid_paint(logpre_excel, grid_size, "gridmap_dif", shp_file)

    stats.ui.progressBar.setValue(5)  # 进度条
    print(kriexcel_list)

    ap.result_processing(kriexcel_list)

    return all_success_file, all_fail_file, suce_excel2, fail_excel2



class XCombobox(QComboBox):
    itemChecked = Signal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.checks = []
        list_widget = QListWidget(self)
        self.setView(list_widget)
        self.setModel(list_widget.model())
        line_edit = QLineEdit(self)
        line_edit.setReadOnly(True)
        self.setLineEdit(line_edit)
        self.add_item('全选')


    def add_item(self, text):
        check = QCheckBox(text, self.view())
        check.stateChanged.connect(self.on_state_changed)
        self.checks.append(check)
        item = QListWidgetItem(self.view())
        item.setFlags(item.flags() & Qt.IntersectsItemShape)
        self.view().addItem(item)
        self.view().setItemWidget(item, check)

    def add_items(self, texts):
        for text in texts:
            self.add_item(text)

    def clear(self):
        self.view().clear()

    def get_selected(self):
        selected_data = []
        for chk in self.checks:
            if self.checks[0] == chk:
                continue
            if chk.checkState() == Qt.Checked:
                selected_data.append(chk.text())
        return selected_data

    def set_all_state(self, state):
        for chk in self.checks:
            chk.blockSignals(True)
            chk.setCheckState(Qt.CheckState(state))
            chk.blockSignals(False)

    def on_state_changed(self, state):
        if self.sender() == self.checks[0]:
            self.set_all_state(state)
        sel_data = self.get_selected()
        self.itemChecked.emit(sel_data)
        self.lineEdit().setText(';'.join(sel_data))


def print_env(data):
    global col
    col = data
    print(data)

class Stats(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('聚类区划.ui')
        self.ui.button.clicked.connect(self.choose_file)
        self.ui.button_2.clicked.connect(self.choose_map)
        self.ui.start.clicked.connect(self.start_button)
        self.ui.progressBar.setRange(0, 5) #进度条
        self.ui.progressBar.reset()

        self.label_2 = self.ui.label_2
        self.image_label = self.ui.image_label
        self.image_label2 = self.ui.image_label2
        self.text_label = self.ui.text_label
        self.column_names_sheet1 = []
        self.column_names_sheet2 = []

        button_pos = self.label_2.geometry().topLeft()
        #self.cmbox = XCombobox(self.ui)
        #self.cmbox.itemChecked.connect(print_soil)
        #self.cmbox.move(130, 100)
        #self.cmbox.setFixedSize(150, 25)

        self.cmbox_env = XCombobox(self.ui)
        self.cmbox_env.itemChecked.connect(print_env)
        self.cmbox_env.move(150,190)
        self.cmbox_env.setFixedSize(234, 21)

        self.soil_box = self.ui.soil_box
        self.soil_box.addItems(['请选择一个重金属', 'item2', ' item3'])
        self.soil_box.setFixedSize(234, 21)

        #self.soil_box.move(243,100)
        self.ui.soil_box.currentIndexChanged.connect(self.comboBox_changed)  # 连接ComboBox的选择变化信号到函数



        self.ui.way1.clicked.connect(lambda: self.onRadioButtonClicked(self.ui.way1.text()))
        self.ui.way2.clicked.connect(lambda: self.onRadioButtonClicked(self.ui.way2.text()))
        self.ui.way3.clicked.connect(lambda: self.onRadioButtonClicked(self.ui.way3.text()))

    def onRadioButtonClicked(self, name):
        global method
        print(f'选中的选项是：{name}')
        method = name
        print(method)


    def comboBox_changed(self):
        self.lab = self.ui.soil_box.currentText()  # 获取ComboBox的当前文本值并赋给 self.lab
        global label
        label = [self.lab]  # 将 self.lab 的值添加到全局的 label 列表中

        print(label)

    def choose_file(self):
        global AfilePath
        AfilePath, _ = QFileDialog.getOpenFileName(self.ui, "选择Excel文件")
        self.ui.text.setText(AfilePath)
        self.load_excel_data()

    def choose_map(self):
        global AmapPath
        AmapPath, _ = QFileDialog.getOpenFileName(self.ui, "选择地图")
        self.ui.text_2.setText(AmapPath)


    def load_excel_data(self):
        try:
            df_sheet1 = pd.read_excel(AfilePath, sheet_name='Sheet1')
            df_sheet2 = pd.read_excel(AfilePath, sheet_name='Sheet2')

            self.column_names_sheet1 = df_sheet1.columns.tolist()
            self.column_names_sheet2 = df_sheet2.columns.tolist()

            #self.cmbox.clear()
            self.cmbox_env.clear()
            self.soil_box.clear()
            self.soil_box.addItems(self.column_names_sheet1)
            #self.cmbox.add_items(self.column_names_sheet1)
            self.cmbox_env.add_items(self.column_names_sheet2)


        except Exception as e:
            print("Error reading Excel file:", str(e))
            #self.cmbox.add_items(['none'])
            self.cmbox_env.add_items(['none'])



    def start_button(self):
        self.window('点击OK开始运行')
        print('----------------------------')
        print(AfilePath)
        run()
        self.load_images()
        self.load_images2()

        self.load_text()
        self.window('运行完毕！')

    def load_text(self):
        # 将列名添加居中对齐标记，然后转化为字符串
        endframe_str = endframe.rename(columns=lambda x: f"{x:^10}").to_string()
        print(endframe_str)
        self.text_label.setText(endframe_str)#

    def load_text2(self, text1, text2, text3, text4):
        stext1 = '\n'.join(text1)
        ftext1 = '\n'.join(text2)
        stext2 = '\n'.join(text3)
        ftext2 = '\n'.join(text4)

        self.text_label.setText(f'first time clustering success:\n{stext1}\nfirst time clustering fail:\n{ftext1}\n'
                                f'second time clustering success:\n{stext2}\nsecond time clustering fail:\n{ftext2}')

    def load_images(self):
        folder_path = './final_result/gridmap.png'  # 请替换为实际的图片路径
        pixmap = QPixmap(folder_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def load_images2(self):
        folder_path = './final_result/gridmap_dif.png'  # 请替换为实际的图片路径
        pixmap = QPixmap(folder_path)
        self.image_label2.setPixmap(pixmap)
        self.image_label2.setScaledContents(True)


    def window(self, string):
        message_box = QMessageBox()
        message_box.setText(string)
        message_box.setWindowTitle("弹窗")
        message_box.setIcon(QMessageBox.Information)
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        message_box.setDefaultButton(QMessageBox.Ok)
        result = message_box.exec_()
        if result == QMessageBox.Ok:
            print("用户点击了确定按钮")
        else:
            sys.exit()
            print("用户点击了取消按钮")



app = QApplication([])
stats = Stats()
stats.ui.show()



#print(stats.choose_file())

app.exec_()

