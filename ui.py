import os
from typing import List
import gradio as gr
import numpy as np
import pandas as pd

from dataprocess import get_average_speed, kringing
from geomap import GeoMap
from imageprocess import img_pca_loading, img_random_walk_process, img_pie_percent, img_line_percent
from regression import RegressionModel
from utils import get_shp_files, get_xlsx_files


def ui():

    shp_files = get_shp_files()
    xlsx_files = get_xlsx_files()

    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                p1_file_name_input = gr.Dropdown(
                    choices=list(xlsx_files.keys()), value="", label="数据文件", scale=3, allow_custom_value=True
                )
                p1_sheet_name_input = gr.Dropdown(choices=[], value="", label="选择数据页", interactive=True, allow_custom_value=True)
                p1_model_input = gr.Dropdown(choices=["Ridge", "Linear", "C"], label="选择拟合模型", scale=3)

            with gr.Row(equal_height=True):
                p1_feature_input = gr.Dropdown(choices=[], label="自变量特征选择", multiselect=True, interactive=True, scale=3)
                p1_target_input = gr.Dropdown(choices=[], label="因变量特征选择", interactive=True, scale=1)
                p1_run_button = gr.Button(value="运行", scale=1)
            with gr.Row(equal_height=True):
                gr.Image(label="结果因果模型")

                with gr.Column():
                    p1_img2_output = gr.Image(label="贡献率分析")
                    gr.Image(label="变化分析影响")
        with gr.Tab("路径解析"):
            gr.Markdown("# 潜在历史污染源及污染路径解析")
            with gr.Row(equal_height=True):
                p2_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), label="数据文件", value="", scale=2)
                p2_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), label="地图文件", value="", scale=2)
                with gr.Column(scale=1):
                    p2_lon = gr.Textbox(label="东经")
                    p2_lat = gr.Textbox(label="北纬")
                p2_run_button = gr.Button(value="运行", scale=1)

            with gr.Row(equal_height=True):
                p2_pca_img = gr.Image(label="相似度分析")
                with gr.Column():
                    p2_grid_img = gr.Image(label="污染源")
                    p2_path_img = gr.Image(label="污染路径")
        with gr.Tab("风险区分析"):
            gr.Markdown("# 土壤重金属未来超标风险区分析")
            with gr.Row(equal_height=True):
                p3_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), value="", label="数据文件")
                p3_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), value="", label="地图文件")

            gr.Markdown("累积趋势分析")
            with gr.Row(equal_height=True):
                p3_time_select = gr.Dropdown([], label="指定年份")
                p3_run_button = gr.Button("运行")
            with gr.Row(equal_height=True):
                p3_dot_img = gr.Image(label="监测点位累积幅度空间分布图")
                with gr.Column():
                    p3_pie_img = gr.Image(label="监测点位累积幅度占比统计")
                    p3_line_img = gr.Image(label="监测数据年际变化")

            gr.Markdown("超标风险区分析")
            with gr.Row(equal_height=True):
                gr.Dropdown(["2020"], label="预测年份")
                gr.Button("计算")
            with gr.Row(equal_height=True):
                p3_speed_output = gr.Textbox(label="年均累计速率(mg/(kg·y)):")
            with gr.Row(equal_height=True):
                gr.Image(label="未来超标风险区空间分布图")

        with gr.Tab("主因和阈值计算"):
            with gr.Row(equal_height=True):
                p4_file_name_input = gr.Dropdown(
                    choices=list(xlsx_files.keys()), value="", label="数据文件", allow_custom_value=True, scale=2
                )
                p4_model_input = gr.Dropdown(choices=["Ridge", "Linear"], label="选择拟合模型", scale=2)
                p4_run_button = gr.Button(value="运行")
            gr.Markdown("归一化模型")
            p4_fitting_img = gr.Image(label="训练集和测试集拟合效果")
            with gr.Row():
                p4_shap_img = gr.Image(label="主因识别")
                p4_analysis_img = gr.Image(label="关键辅因分析")
            gr.Markdown("安全阈值计算")
            with gr.Row():
                p4_prediction_img = gr.Image(label="预测曲线")
                p4_shiki_img = gr.Image(label="shiki")
            p4_threshold_output = gr.Textbox(label="安全阈值计算")

        # page1

        p1_params: dict = {}

        p1_file_name_input.change(
            fn=lambda file_name: gr.update(choices=list(pd.read_excel(xlsx_files[file_name], sheet_name=None).keys())),
            inputs=p1_file_name_input,
            outputs=p1_sheet_name_input,
        )

        def p1_sheet_change(file_name: str, sheet_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            p1_params["data"] = data
            update = gr.update(choices=data.columns.to_list())
            return update, update

        p1_sheet_name_input.change(
            fn=p1_sheet_change, inputs=[p1_file_name_input, p1_sheet_name_input], outputs=[p1_target_input, p1_feature_input]
        )

        def p1_run_click(model: str, feature: List[str], target: str):
            model = RegressionModel(model, p1_params["data"], feature, target)
            model.train_and_evaluate_model()
            img_path = model.plot_coefficients()
            return img_path

        p1_run_button.click(fn=p1_run_click, inputs=[p1_model_input, p1_feature_input, p1_target_input], outputs=p1_img2_output)

        # page2
        def p2_run_click(file_name: str, map_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name="原始数据")
            gmap = GeoMap(shp_files[map_name])
            gmap.load_dots_df(data)
            kringing(data, num=50)
            gmap.load_dots_df(pd.read_excel(os.path.join(".", "kringing", "data.xlsx")))
            gmap.grid_paint(20, 20)
            pca_path = img_pca_loading(data)
            img_grid_pure_path = gmap.save_grid_image_pure("Cd")
            img_random_walk_path = img_random_walk_process(img_grid_pure_path)
            return (
                gmap.lon,
                gmap.lat,
                pca_path,
                img_grid_pure_path,
                img_random_walk_path,
            )

        p2_run_button.click(
            fn=p2_run_click, inputs=[p2_file_name_input, p2_map_name_input], outputs=[p2_lon, p2_lat, p2_pca_img, p2_grid_img, p2_path_img]
        )

        # page3

        def p3_file_change(file_name: str):
            data = pd.read_excel(xlsx_files[file_name])
            return gr.update(choices=[item for item in data.columns if item.endswith("年")])

        p3_file_name_input.change(fn=p3_file_change, inputs=p3_file_name_input, outputs=p3_time_select)

        def p3_run_click(file_name: str, map_name: str, label: str):
            data = pd.read_excel(xlsx_files[file_name])
            gmap = GeoMap(shp_files[map_name])
            gmap.load_dots_df(
                data,
                params_name=[
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
                ],
            )
            img_dot_path = gmap.save_dot_image(label)
            img_pie_path = img_pie_percent(data, label)
            img_line_path = img_line_percent(data)
            average_speed = get_average_speed(data)

            return img_dot_path, img_pie_path, img_line_path, average_speed

        p3_run_button.click(
            fn=p3_run_click,
            inputs=[
                p3_file_name_input,
                p3_map_name_input,
                p3_time_select,
            ],
            outputs=[
                p3_dot_img,
                p3_pie_img,
                p3_line_img,
                p3_speed_output,
            ],
        )

        # page4

        def p4_run_click(file_name: str, model_name: str):
            data = pd.read_excel(xlsx_files[file_name])
            feature = ["Cd", "Pb", "Al", "Ca", "Mn"]
            target = "水稻Cd"
            model = RegressionModel(model_name, data, feature, target)
            model.train_and_evaluate_model()
            img_fitting_path = model.plot_fitting_effect()
            img_shap_path = model.plot_shap_importance()

            base_feature_values = {"Pb": 43.4, "Al": 10.8, "Ca": 747.2, "Mn": 240.2}

            # 计算土壤 Cd 的安全阈值
            threshold_cd = model.calculate_threshold(fixed_values=base_feature_values, variable_feature="Cd", target_value=0.2)
            variable_range = np.linspace(0, 2, 100)
            img_prediction_path = model.plot_prediction_curve(
                fixed_values=base_feature_values,
                variable_feature="Cd",
                variable_range=variable_range,
            )

            return img_fitting_path, img_shap_path, img_prediction_path, threshold_cd

        p4_run_button.click(
            fn=p4_run_click,
            inputs=[p4_file_name_input, p4_model_input],
            outputs=[p4_fitting_img, p4_shap_img, p4_prediction_img, p4_threshold_output],
        )

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
