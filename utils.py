import os
import glob
from typing import List


def get_files(dict_path: str, extension: str):
    # 检查目录是否存在
    if not os.path.exists(dict_path):
        raise ValueError("指定的目录不存在，请检查路径是否正确！")

    files: List[str] = glob.glob(os.path.join(dict_path, f"*.{extension}"))

    # 提取文件名（不包括路径）
    file_dict = {os.path.basename(file): file for file in files}
    return file_dict


def get_xlsx_files():
    return get_files(os.path.join(".", "数据"), "xlsx")


def get_shp_files():
    return get_files(os.path.join(".", "地图"), "shp")
