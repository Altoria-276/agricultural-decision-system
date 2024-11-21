import gradio as gr


def ui():
    with gr.Blocks() as ui:
        gr.Markdown("## “污染源—土壤—活化—作物”全链条风险预警模拟器")
        with gr.Tab("影响分析"):
            gr.Markdown("# 现存污染源贡献量变化影响分析")
            with gr.Row(equal_height=True):
                gr.File(label="文件路径", scale=3)
                gr.Dropdown(choices=["A", "B", "C"], label="选择拟合模型", scale=3)
                gr.Button(value="运行", scale=1)
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

    ui.launch(share=False)


if __name__ == "__main__":
    ui()
