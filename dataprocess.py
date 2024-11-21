import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from alphashape import alphashape
from shapely.geometry import Polygon, MultiPolygon, Point



"""克里金插值"""
def kringing(self, df, attributes, longitude, latitude, file_name,  outpath, num):
    # 读取指定范围的shapefile文件
    #specified_shapefile = gpd.read_file(maskBoundary)

    # 获取经度和纬度的最大最小值
    #min_longitude = specified_shapefile.bounds['minx'].min()
    #max_longitude = specified_shapefile.bounds['maxx'].max()
    #min_latitude = specified_shapefile.bounds['miny'].min()
    #max_latitude = specified_shapefile.bounds['maxy'].max()

    print(' ')
    min_longitude, max_longitude = df[space[0]].min(), df[space[0]].max()
    min_latitude, max_latitude = df[space[1]].min(), df[space[1]].max()

    # 创建一个空GeoDataFrame用于存储插值结果
    geometry = [Point(xy) for xy in zip(np.tile(np.linspace(min_longitude, max_longitude, num), num),
                                         np.repeat(np.linspace(min_latitude, max_latitude, num), num))]
    crs = {'init': 'epsg:4326'}  # EPSG:4326坐标系
    interpolated_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs=crs, geometry=geometry)

    # 循环遍历每个属性进行插值
    for attribute in attributes:
        data = df[attribute]
        OK = OrdinaryKriging(longitude, latitude, data, variogram_model='spherical', nlags=3)
        z, ss = OK.execute('grid', np.linspace(min_longitude, max_longitude, num), np.linspace(min_latitude, max_latitude, num))
        interpolated_gdf[attribute] = z.flatten()

    # 保存插值结果为shapefile文件和数据表
    interpolated_data_shp_path = f"./{outpath}/{file_name}.shp"
    interpolated_gdf.to_file(interpolated_data_shp_path)

    # 提取经度和纬度信息
    interpolated_gdf['经度'] = interpolated_gdf.geometry.apply(lambda geom: geom.x)
    interpolated_gdf['纬度'] = interpolated_gdf.geometry.apply(lambda geom: geom.y)

    # 将经度和纬度列放在前两列
    interpolated_gdf = interpolated_gdf[['经度', '纬度'] + attributes]

    interpolated_data_xlsx_path = f"./{outpath}/{file_name}.xlsx"
    interpolated_gdf.to_excel(interpolated_data_xlsx_path, index=False, engine='xlsxwriter')
    print("插值数据保存成功！")
    print('文件名称')
    print(f"./{outpath}/{file_name}.xlsx")
    return f"./{outpath}", f"./{outpath}/{file_name}.xlsx"

"""散点与SHP映射"""
def poi2image(self,all_file, cluster_folder, excel_name, outfile_path):
    alldata = pd.read_excel(all_file)

    # 创建绘图对象
    fig, ax = plt.subplots()
    # 绘制shp边界
    data.plot(ax=ax, facecolor='none', edgecolor='black')
    m = 1
    c = 1
    handles = []
    labels = []
    for j in excel_name:
        df = pd.read_excel(f"./{cluster_folder}/{j}")

        # 经度和纬度数据
        longitude = df.loc[:, '经度']
        latitude = df.loc[:, '纬度']

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
            #boundary_x = [x for x, y in boundary.exterior.coords]
            #boundary_y = [y for x, y in boundary.exterior.coords]
            # 使用B-spline曲线拟合
            #tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
            # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
            #u_new = np.linspace(u.min(), u.max(), 1000)
            #x_new, y_new = interpolate.splev(u_new, tck)
            # 绘制平滑后的曲线
            #plt.plot(x_new, y_new, color=colors[j])
            ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
            handles.append(ax.scatter([], [], color=colors[c]))
            # labels.append(j)
            gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])
            gdf.crs = 'EPSG:4326'

            # 设置输出文件的路径和名称
            output_file_name = 'boundary'+str(m)+'.shp'
            outfilepath = f"./{outfile_path}/{output_file_name}"
            m = m + 1

            # 将GeoDataFrame保存为shp文件
            gdf.to_file(outfilepath, driver='ESRI Shapefile')
            df = pd.DataFrame()

            for a in alldata.index:
                point = Point(alldata.loc[a, '经度'], alldata.loc[a, '纬度'])
                if point.within(boundary):
                    df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
                    df = pd.concat([df, df1], axis=0, ignore_index=True)
            if len(df) > 20:
                excel_name_out = '区域' + str(m - 1) + '.xlsx'
                labels_name = '区域' + str(m - 1)
                labels.append(labels_name)
                excel_path_out = f"./{outfile_path}/{excel_name_out}"
                df.to_excel(excel_path_out, index=False)
            else:
                m=m-1

        elif isinstance(boundary, MultiPolygon):
            # 多个多边形边界
            # 绘制多边形
            for polygon in boundary.geoms:
                #boundary_x = [x for x, y in polygon.exterior.coords]
                #boundary_y = [y for x, y in polygon.exterior.coords]
                # 使用B-spline曲线拟合
                #tck, u = interpolate.splprep([boundary_x, boundary_y], s=0)
                # 定义新的曲线参数，增加插值点的数量以获得平滑的曲线
                #u_new = np.linspace(u.min(), u.max(), 1000)
                #x_new, y_new = interpolate.splev(u_new, tck)
                # 绘制平滑后的曲线
                #plt.plot(x_new, y_new, color=colors[j])
                expanded_polygon = polygon.buffer(0.003)
                x, y = expanded_polygon.exterior.xy
                ax.plot(x, y, color=colors[c], label=j)  # 添加label参数
                handles.append(ax.scatter([], [], color=colors[c]))
                # labels.append(j)
                #print(expanded_polygon)
                gdf = gpd.GeoDataFrame(geometry=[expanded_polygon])

                gdf.crs = 'EPSG:4326'
                # 设置输出文件的路径和名称
                output_file_name = 'boundary'+str(m)+'.shp'
                outfilepath = f"./{outfile_path}/{output_file_name}"
                m=m+1
                # 将GeoDataFrame保存为shp文件
                gdf.to_file(outfilepath, driver='ESRI Shapefile')

                df = pd.DataFrame()
                for a in alldata.index:
                    point = Point(alldata.loc[a, '经度'], alldata.loc[a, '纬度'])
                    if point.within(polygon):
                        df1 = pd.DataFrame(alldata.loc[a]).T  # 将Series转换为DataFrame
                        df = pd.concat([df, df1], axis=0, ignore_index=True)

                if len(df) > 20:
                    excel_name_out = '区域'+str(m-1)+'.xlsx'
                    labels_name = '区域' + str(m - 1)
                    labels.append(labels_name)
                    excel_path_out = f"./{outfile_path}/{excel_name_out}"
                    df.to_excel(excel_path_out, index=False)
                else:
                    m = m - 1

        # 绘制边界
        #plt.plot(*boundary.exterior.xy, color=colors[j])
        # 绘制散点图
        plt.scatter(longitude, latitude, color=colors[c], s=1)
        c = c + 1
    # 绘制多边形
    #data.plot(ax=ax, facecolor='none', edgecolor='black')
    data.plot(ax=ax, facecolor='none', edgecolor='black')
    handles = list(set(handles))
    labels = list(set(labels))
    plt.legend(handles, labels, loc='upper left')

    # 设置坐标轴范围
    ax.set_xlim(data.total_bounds[0]-0.2, data.total_bounds[2]+0.01)
    ax.set_ylim(data.total_bounds[1]-0.01, data.total_bounds[3]+0.01)

    plt.axis('off')  # 不显示坐标轴
    # 设置图形属性
    #ax.set_aspect('equal')  # 保持纵横比相等

    plt.savefig(f'./final_result/过程{m-1}.png')
    # 显示图形
    #plt.show()
    return m-1


"""栅格化"""
def grid_paint(kriexcel_list, grid_size, image_name, shp_file):

    # 创建图形和轴对象
    fig, ax = plt.subplots()

    i = 1
    for excelname,shapefile in zip(kriexcel_list, shp_file):

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

        i = i+1
        # 绘制栅格图，并获取返回的图像对象
        img = ax.imshow(grid, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', alpha=0.5)

        # 绘制Shapefile的边界
        shapefile.boundary.plot(ax=ax, color='red', linewidth=1)
        # 显示图形并保存图像文件
        # 添加颜色条

    # 绘制地图边界
    data = gpd.read_file(AmapPath)
    data.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(img, ax=ax)
    #plt.show()
    save_path = './final_result/'+image_name+'.png'
    plt.savefig(save_path)

