import os
import gradio as gr
import pandas as pd

from dataprocess import kringing
from geomap import GeoMap
from imageprocess import img_pca_loading, img_random_walk_process
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
                p1_model_input = gr.Dropdown(
                    choices=["Ridge", "Linear", "C"], value="", label="选择拟合模型", scale=3, allow_custom_value=True
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
                p2_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), label="数据文件", scale=2)
                p2_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), label="地图文件", scale=2)
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
                p3_file_name_input = gr.Dropdown(choices=list(xlsx_files.keys()), label="数据文件")
                p3_map_name_input = gr.Dropdown(choices=list(shp_files.keys()), label="地图文件")
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("累积趋势分析")
                    with gr.Row(equal_height=True):
                        p3_time_select = gr.Dropdown(["2008"], label="指定年份")
                        p3_run_button = gr.Button("运行")
                    with gr.Row(equal_height=True):
                        gr.Image(label="监测点位累积幅度空间分布图")
                        with gr.Column():
                            gr.Image(label="监测点位累积幅度占比统计")
                            gr.Image(label="监测数据年际变化")

                with gr.Column():
                    gr.Markdown("超标风险区分析")
                    with gr.Row(equal_height=True):
                        gr.Dropdown(["2020"], label="预测年份")
                        gr.Button("计算")
                    with gr.Row(equal_height=True):
                        gr.Image(label="未来超标风险区空间分布图")

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

        def p1_model_change(model):
            p1_params["model"] = model

        p1_model_input.change(fn=p1_model_change, inputs=p1_model_input)

        def p1_target_change(target):
            p1_params["target"] = target

        p1_target_input.change(fn=p1_target_change, inputs=p1_target_input)

        def p1_feature_change(feature):
            p1_params["feature"] = feature

        p1_feature_input.change(fn=p1_feature_change, inputs=p1_feature_input)

        def p1_run_click():
            model = RegressionModel(p1_params["model"], p1_params["data"], p1_params["feature"], p1_params["target"])
            model.train_and_evaluate_model()
            img_path = model.plot_coefficients()
            return img_path

        p1_run_button.click(fn=p1_run_click, outputs=p1_img2_output)

        # page2
        def p2_run_click(file_name: str, map_name: str):
            data = pd.read_excel(xlsx_files[file_name], sheet_name="原始数据")
            gmap = GeoMap(shp_files[map_name])
            gmap.load_dots_df(data)
            kringing(data, num=100)
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

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
