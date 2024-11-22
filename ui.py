from random import choice, choices
from turtle import update
import gradio as gr
import pandas as pd


def ui():
    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                file_path_input = gr.File(label="文件路径", scale=3)
                sheet_name_input = gr.Dropdown(choices=["无"], value="无", label="选择数据页", interactive=True)
                model_input = gr.Dropdown(choices=["Ridge", "Linear", "C"], label="选择拟合模型", scale=3)

            with gr.Row(equal_height=True):
                feather_input = gr.Dropdown(choices=[], label="自变量特征选择", multiselect=True, interactive=True)
                target_input = gr.Dropdown(choices=[], label="因变量特征选择", interactive=True)
                run_button = gr.Button(value="运行", scale=1)
            with gr.Row(equal_height=True):
                gr.Image(label="结果因果模型")

                with gr.Column():
                    gr.Image(label="贡献率分析")
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
                    gr.File(label="基准数据")
                    gr.File(label="监测数据")
                with gr.Column():
                    gr.File(label="行政区划地图文件")
                    gr.File(label="土地利用现状图")
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("累积趋势分析")
                    with gr.Row(equal_height=True):
                        gr.Dropdown(["2020"], label="指定年份")
                        gr.Button("计算")
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

        file_path_input.change(
            fn=lambda file_path: gr.update(choices=pd.ExcelFile(file_path).sheet_names),
            inputs=file_path_input,
            outputs=sheet_name_input,
        )

        def sheet_change(file_path: str, sheet_name: str):
            data = pd.ExcelFile(file_path).parse(sheet_name)
            update = gr.update(choices=data.columns.to_list())
            return update, update

        sheet_name_input.change(fn=sheet_change, inputs=[file_path_input, sheet_name_input], outputs=[target_input, feather_input])

        # target_input.change(
        #     fn=lambda file_path, sheet_name, target_name: gr.update(
        #         choices=[col for col in pd.ExcelFile(file_path).parse(sheet_name).columns.to_string() if col != target_name]
        #     ),
        #     inputs=[file_path_input, sheet_name_input, target_input],
        #     outputs=feather_input,
        # )

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
