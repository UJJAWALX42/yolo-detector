from ultralytics import YOLO
import gradio as gr

# Load model
model = YOLO("best.pt")

def detect(image):
    results = model.predict(image, conf=0.25)
    annotated = results[0].plot()
    return annotated

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Detection Result"),
    title="YOLO11 Object Detection",
    description="Upload an image to detect objects."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")