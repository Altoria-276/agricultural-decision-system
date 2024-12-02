from typing import List
import gradio as gr
import numpy as np
import pandas as pd

from dataprocess import get_average_speed, kringing
from geomap import GeoMap
from imageprocess import img_pca_loading, img_random_walk_process, img_pie_percent, img_line_percent
from regression import RegressionModel
from utils import get_model_choices, get_shp_files, get_xlsx_files


def ui():

    shp_files = get_shp_files()
    xlsx_files = get_xlsx_files()

    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                p1_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), value="", label="数据文件", allow_custom_value=True)
                p1_sheet_name_input = gr.Dropdown(choices=[], value="", label="选择数据页", interactive=True, allow_custom_value=True)
                p1_model_input = gr.Dropdown(
                    choices=get_model_choices(),
                    label="选择拟合模型",
                )

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
                with gr.Column(scale=6):
                    with gr.Row():
                        p2_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), label="地图文件", scale=2)
                        p2_file_name_input = gr.Dropdown(
                            choices=list(xlsx_files.keys()), value="", label="数据文件", scale=2, allow_custom_value=True
                        )
                        p2_sheet_name_input = gr.Dropdown(
                            choices=[], value="", label="选择数据页", interactive=True, allow_custom_value=True
                        )
                    with gr.Row(equal_height=True):
                        p2_params_input = gr.Dropdown(choices=[], label="特征选择", multiselect=True, interactive=True, scale=3)
                        with gr.Column():
                            p2_label_input = gr.Dropdown(choices=[], label="污染源特征选择", interactive=True, scale=1)
                            p2_n_pca_input = gr.Dropdown(choices=[0], value=0, label="主因数量选择")
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
                p3_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), value="", label="数据文件", allow_custom_value=True)
                p3_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), label="地图文件")

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
            gr.Markdown("# 土壤重金属活化主因及安全阈值测算")
            with gr.Row(equal_height=True):
                p4_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), label="数据文件")
                p4_sheet_name_input = gr.Dropdown(choices=[], value="", label="选择数据页", interactive=True, allow_custom_value=True)
                p4_model_input = gr.Dropdown(choices=get_model_choices(), label="选择拟合模型")
            with gr.Row(equal_height=True):
                p4_feature_input = gr.Dropdown(
                    choices=[], label="自变量特征选择(至少选择 5 个)", multiselect=True, interactive=True, scale=3
                )
                p4_target_input = gr.Dropdown(choices=[], label="因变量特征选择", interactive=True, scale=1)
                p4_run_button = gr.Button(value="运行", scale=1)
            gr.Markdown("归一化模型")
            p4_fitting_img = gr.Image(label="训练集和测试集拟合效果")
            with gr.Row():
                p4_shap_img = gr.Image(label="主因识别")
                p4_hassian_img = gr.Image(label="关键辅因分析")
            gr.Markdown("安全阈值计算")
            with gr.Row(equal_height=True):
                p4_factor1 = gr.Textbox(label="factor1")
                p4_factor2 = gr.Textbox(label="factor2")
                p4_factor3 = gr.Textbox(label="factor3")
                p4_factor4 = gr.Textbox(label="factor4")
                p4_factor5 = gr.Textbox(label="factor5")
                p4_run_button_2 = gr.Button("运行")
            with gr.Row():
                p4_prediction_img = gr.Image(label="预测曲线")
                p4_shiki_img = gr.Image(label="shiki")
            p4_threshold_output = gr.Textbox(label="安全阈值计算")
        with gr.Tab("区域分布预测"):
            gr.Markdown("# 土壤重金属活化主因及安全阈值测算")
            with gr.Row(equal_height=True):
                p5_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), label="数据文件")
                p5_sheet_name_input = gr.Dropdown(choices=[], value="", label="选择数据页", interactive=True, allow_custom_value=True)
                p5_model_input = gr.Dropdown(choices=get_model_choices(), label="选择拟合模型")
            with gr.Row(equal_height=True):
                p5_feature_input = gr.Dropdown(choices=[], label="自变量特征选择", multiselect=True, interactive=True, scale=3)
                p5_target_input = gr.Dropdown(choices=[], label="因变量特征选择", interactive=True, scale=1)
                p5_run_button = gr.Button(value="运行", scale=1)

        # page1
        p1_file_name_input.change(
            fn=lambda file_name: gr.update(choices=list(pd.read_excel(xlsx_files[file_name], sheet_name=None).keys())),
            inputs=p1_file_name_input,
            outputs=p1_sheet_name_input,
        )

        def p1_sheet_change(file_name: str, sheet_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            update = gr.update(choices=data.columns.to_list())
            return update, update

        p1_sheet_name_input.change(
            fn=p1_sheet_change, inputs=[p1_file_name_input, p1_sheet_name_input], outputs=[p1_target_input, p1_feature_input]
        )

        def p1_run_click(file_name: str, sheet_name: str, model_name: str, feature: List[str], target: str):
            data = pd.read_excel(get_xlsx_files()[file_name], sheet_name)
            model = RegressionModel(model_name, data, feature, target)
            model.train_and_evaluate_model()
            img_path = model.plot_coefficients()
            return img_path

        p1_run_button.click(
            fn=p1_run_click,
            inputs=[
                p1_file_name_input,
                p1_sheet_name_input,
                p1_model_input,
                p1_feature_input,
                p1_target_input,
            ],
            outputs=p1_img2_output,
        )

        # page2
        p2_file_name_input.change(
            fn=lambda file_name: gr.update(choices=list(pd.read_excel(xlsx_files[file_name], sheet_name=None).keys())),
            inputs=p2_file_name_input,
            outputs=p2_sheet_name_input,
        )

        def p2_sheet_change(file_name: str, sheet_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            update = gr.update(choices=data.columns.to_list())
            return update

        p2_sheet_name_input.change(fn=p2_sheet_change, inputs=[p2_file_name_input, p2_sheet_name_input], outputs=p2_params_input)

        p2_params_input.change(
            fn=lambda x: (gr.update(choices=x), gr.update(choices=[_ for _ in range(len(x))])),
            inputs=p2_params_input,
            outputs=[p2_label_input, p2_n_pca_input],
        )

        def p2_run_click(file_name: str, map_name: str, params_name: List[str], label: str, n: int):
            data = pd.read_excel(xlsx_files[file_name], sheet_name="原始数据")
            gmap = GeoMap(shp_files[map_name])
            gmap.load_dots_df(data, params_name=params_name)
            kringing_path = kringing(data, params_name=params_name, num=50)
            gmap.load_dots_df(pd.read_excel(kringing_path), params_name=params_name)
            gmap.grid_paint(20, 20)
            pca_path = img_pca_loading(data, params_name, n)
            img_grid_grey_path = gmap.save_grid_image_grey(label)
            img_grid_path = gmap.save_grid_image(label)
            img_random_walk_path = img_random_walk_process(img_grid_path, img_grid_grey_path)
            return (
                gmap.lon,
                gmap.lat,
                pca_path,
                img_grid_path,
                img_random_walk_path,
            )

        p2_run_button.click(
            fn=p2_run_click,
            inputs=[
                p2_file_name_input,
                p2_map_name_input,
                p2_params_input,
                p2_label_input,
                p2_n_pca_input,
            ],
            outputs=[p2_lon, p2_lat, p2_pca_img, p2_grid_img, p2_path_img],
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
                params_name=list(data.columns[3:]),
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

        p4_file_name_input.change(
            fn=lambda file_name: gr.update(choices=list(pd.read_excel(xlsx_files[file_name], sheet_name=None).keys())),
            inputs=p4_file_name_input,
            outputs=p4_sheet_name_input,
        )

        def p4_sheet_change(file_name: str, sheet_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            update = gr.update(choices=data.columns.to_list())
            return update, update

        p4_sheet_name_input.change(
            fn=p4_sheet_change, inputs=[p4_file_name_input, p4_sheet_name_input], outputs=[p4_target_input, p4_feature_input]
        )

        def p4_run_click(file_name: str, sheet_name: str, model_name: str, feature: List[str], target: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            model = RegressionModel(model_name, data, feature, target)
            model.train_and_evaluate_model()
            img_fitting_path = model.plot_fitting_effect()
            img_shap_path = model.plot_shap_importance()
            img_hessian_path = model.plot_hessian_matrix()
            top_features = model.get_top_feature()

            update1, update2, update3, update4, update5 = [
                gr.update(label=top_feature[0], value=top_feature[1]) for top_feature in top_features
            ]

            return (
                img_fitting_path,
                img_shap_path,
                img_hessian_path,
                top_features,
                update1,
                update2,
                update3,
                update4,
                update5,
            )

        p4_params = gr.State()

        p4_run_button.click(
            fn=p4_run_click,
            inputs=[
                p4_file_name_input,
                p4_sheet_name_input,
                p4_model_input,
                p4_feature_input,
                p4_target_input,
            ],
            outputs=[
                p4_fitting_img,
                p4_shap_img,
                p4_hassian_img,
                p4_params,
                p4_factor1,
                p4_factor2,
                p4_factor3,
                p4_factor4,
                p4_factor5,
            ],
        )

        def p4_run_click_2(
            file_name: str,
            sheet_name: str,
            model_name: str,
            target: str,
            params,
            factor1: float,
            factor2: float,
            factor3: float,
            factor4: float,
            factor5: float,
        ):
            data = pd.read_excel(xlsx_files[file_name], sheet_name)
            params[0][1] = factor1
            params[1][1] = factor2
            params[2][1] = factor3
            params[3][1] = factor4
            params[4][1] = factor5
            feature = [params[0][0], params[1][0], params[2][0], params[3][0], params[4][0]]
            if "Cd" not in feature:
                feature.append("Cd")
            model = RegressionModel(model_name, data, feature, target)
            model.train_and_evaluate_model()
            base_feature_values = dict(params)

            # 计算土壤 Cd 的安全阈值
            threshold_cd = model.calculate_threshold(fixed_values=base_feature_values, variable_feature="Cd", target_value=0.2)
            variable_range = np.linspace(0, 2, 100)
            img_prediction_path = model.plot_prediction_curve(
                fixed_values=base_feature_values,
                variable_feature="Cd",
                variable_range=variable_range,
            )

            return threshold_cd, img_prediction_path

        p4_run_button_2.click(
            fn=p4_run_click_2,
            inputs=[
                p4_file_name_input,
                p4_sheet_name_input,
                p4_model_input,
                p4_target_input,
                p4_params,
                p4_factor1,
                p4_factor2,
                p4_factor3,
                p4_factor4,
                p4_factor5,
            ],
            outputs=[
                p4_threshold_output,
                p4_prediction_img,
            ],
        )

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
