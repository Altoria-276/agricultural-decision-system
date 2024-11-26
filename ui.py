import gradio as gr
import pandas as pd

from geomap import GeoMap
from imageprocess import img_pca_loading, img_random_walk_process
from regression import RegressionModel


def ui():
    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                p1_file_path_input = gr.File(label="文件路径", scale=3)
                p1_sheet_name_input = gr.Dropdown(choices=["无"], value="无", label="选择数据页", interactive=True)
                p1_model_input = gr.Dropdown(choices=["Ridge", "Linear", "C"], value="", label="选择拟合模型", scale=3)

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
                p2_file_path_input = gr.File(label="数据文件路径", scale=2)
                p2_map_path_input = gr.File(label="地图文件路径", scale=2)
                with gr.Column(scale=1):
                    p2_lon = gr.Textbox(label="东经")
                    p2_lat = gr.Textbox(label="北纬")
                p2_run_button = gr.Button(value="运行", scale=1)

            with gr.Row(equal_height=True):
                p2_pca_img = gr.Image(label="相似度分析")
                p2_grid_img = gr.Image(label="污染源")
                p2_path_img = gr.Image(label="污染路径")
        with gr.Tab("风险区分析"):
            gr.Markdown("# 土壤重金属未来超标风险区分析")
            with gr.Row(equal_height=True):
                with gr.Column():
                    p3_file_data_input = gr.File(label="数据")
                with gr.Column():
                    p3_file_map_input = gr.File(label="地图文件")
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

        p1_file_path_input.change(
            fn=lambda file_path: gr.update(choices=list(pd.read_excel(file_path, sheet_name=None).keys())),
            inputs=p1_file_path_input,
            outputs=p1_sheet_name_input,
        )

        def p1_sheet_change(file_path: str, sheet_name: str):
            data = pd.read_excel(file_path, sheet_name)
            p1_params["data"] = data
            update = gr.update(choices=data.columns.to_list())
            return update, update

        p1_sheet_name_input.change(
            fn=p1_sheet_change, inputs=[p1_file_path_input, p1_sheet_name_input], outputs=[p1_target_input, p1_feature_input]
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

        p2_params: dict = {}

        def p2_file_change(file_path: str):
            data = pd.read_excel(file_path, "原始数据")
            p2_params["data"] = data

        p2_file_path_input.change(fn=p2_file_change)

        def p2_map_change(file_path: str):
            p2_params["file_path"] = file_path

        p2_map_path_input.change(fn=p2_map_change)

        def p2_run_click():
            gmap = GeoMap(p2_params["file_path"], p2_params["data"])
            pca_path = img_pca_loading(p2_params["data"])
            img_grid_pure_path = gmap.save_grid_image_pure("Cd")
            img_random_walk_path = img_random_walk_process(img_grid_pure_path)
            return (
                gmap.lon,
                gmap.lat,
                pca_path,
                img_grid_pure_path,
                img_random_walk_path,
            )

        p2_run_button.click(fn=p2_run_click, outputs=[p2_lon, p2_lat, p2_pca_img, p2_grid_img, p2_path_img])

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
