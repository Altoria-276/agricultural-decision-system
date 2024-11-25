import gradio as gr
import pandas as pd

from regression import RegressionProcess


def ui():
    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                p1_file_path_input = gr.File(label="文件路径", scale=3)
                p1_sheet_name_input = gr.Dropdown(choices=["无"], value="无", label="选择数据页", interactive=True)
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
                gr.File(label="文件路径", scale=3)
                with gr.Column(scale=1):
                    gr.Textbox(label="东经")
                    gr.Textbox(label="北纬")
                gr.Button(value="运行", scale=1)

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("相似度分析")
                    gr.Image()
                    gr.Image()
                gr.Image(label="污染源及污染路径")
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

        regression_model = RegressionProcess()

        p1_file_path_input.change(
            fn=lambda file_path: gr.update(choices=pd.ExcelFile(file_path).sheet_names),
            inputs=p1_file_path_input,
            outputs=p1_sheet_name_input,
        )

        def sheet_change(file_path: str, sheet_name: str):
            data = pd.ExcelFile(file_path).parse(sheet_name)
            update = gr.update(choices=data.columns.to_list())
            regression_model.load_model(p1_model_input.value)
            regression_model.data = data
            return update, update

        p1_sheet_name_input.change(
            fn=sheet_change, inputs=[p1_file_path_input, p1_sheet_name_input], outputs=[p1_target_input, p1_feature_input]
        )

        def target_change(target: str):
            if regression_model:
                regression_model.target = target

        p1_target_input.change(fn=target_change, inputs=p1_target_input)

        def feature_change(feature):
            if regression_model:
                regression_model.feature = feature

        p1_feature_input.change(fn=feature_change, inputs=p1_feature_input)

        p1_run_button.click(fn=regression_model.run, outputs=p1_img2_output)

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
