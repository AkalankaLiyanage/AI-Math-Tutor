import gradio as gr
from model import generate_response

def math_tutor(question):
    return generate_response(question)

iface = gr.Interface(fn=math_tutor, inputs="text", outputs="text", title="AI Math Tutor")
iface.launch()
