from ImageProcess import *

# img_random_walk_process("Images/img_grid_Cd_pure.png")
# img_random_walk_process("Images/img_grid_Fe_pure.png")
# img_random_walk_process("Images/img2.png")


# 示例调用
# file_path = "数据/水稻点位148.xlsx"  # Excel文件路径
# sheet_name = "原始数据"  # Excel表单名称
# params_name = [
#     "P",
#     "K",
#     "N",
#     "Cr",
#     "Cu",
#     "Zn",
#     "As",
#     "Cd",
#     "Pb",
#     "Se",
#     "Mo",
#     "Na",
#     "Al",
#     "Si",
#     "Ca",
#     "Fe",
#     "Hg",
#     "La",
#     "Mg",
#     "Mn",
#     "有效态Cd",
# ]
#
# # 读取Excel数据
# data = pd.read_excel(file_path, sheet_name=sheet_name)
#
# # 执行 PCA 并降到8个主成分
# pca_result = img_pca_loading(data, params_name, n_components=8)


# 文件路径和表单名称
file_path = "数据/水稻点位148.xlsx"
sheet_name = "原始数据"

# 读取数据
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 分析的因素列表
params_name = [
    "P", "K", "N", "Cr", "Cu", "Zn", "As", "Cd", "Pb", "Se", "Mo", "Na",
    "Al", "Si", "Ca", "Fe", "Hg", "La", "Mg", "Mn", "有效态Cd"
]

# 保存图像的路径
save_path = "Images/方差分析柱状图.png"

# 调用优化后的函数
anova_and_plot(data, params_name)
plot_correlation_matrix(data, params_name)

