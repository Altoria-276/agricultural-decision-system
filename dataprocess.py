import os
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon, Point

from geomap import GeoMap


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


def kringing(df: pd.DataFrame, params_name: List[str] = params_name, num=100):
    outpath = os.path.join(".", "kringing", "data")
    lon, lat = df[space[0]], df[space[1]]

    min_lon, max_lon = lon.min(), lon.max()
    min_lat, max_lat = lat.min(), lat.max()

    lon, lat = df[space[0]], df[space[1]]

    # 创建一个空GeoDataFrame用于存储插值结果
    geometry = [
        Point(xy) for xy in zip(np.tile(np.linspace(min_lon, max_lon, num), num), np.repeat(np.linspace(min_lat, max_lat, num), num))
    ]
    interpolated_gdf = gpd.GeoDataFrame(pd.DataFrame(), crs=CRS("EPSG:4326"), geometry=geometry)

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


def get_average_speed(data: pd.DataFrame):
    se = data.mean()[3:]
    res = (se["2024年"] - se["2008年"]) / (2024 - 2008)
    return res
