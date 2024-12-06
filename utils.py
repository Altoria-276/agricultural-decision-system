import os
import glob
from typing import List


def get_files(dict_path: str, extension: str):
    """
    获取指定目录下的所有指定扩展名的文件

    Args:
        dict_path (str): 目录路径
        extension (str): 扩展名

    Raises:
        ValueError: 指定的目录不存在，请检查路径是否正确！

    Returns:
        dict: 文件路径字典
    """
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


def get_model_choices():
    return [
        "Ridge",
        "Linear",
        "Lasso",
        "SVR",
        "DecisionTree",
        "RandomForest",
    ]


def get_temp_image_path():
    return os.path.join(".", "Images", "Temp")
