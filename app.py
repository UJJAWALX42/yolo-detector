from ultralytics import YOLO
import gradio as gr
import cv2
import os
import tempfile
import imageio

model = YOLO("best.pt")

def detect_image(image):
    results = model.predict(image, conf=0.25)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        writer.append_data(annotated)

    cap.release()
    writer.close()
    return out_path

image_interface = gr.Interface(
    fn=detect_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Detection Result"),
    title="Image Detection",
)

video_interface = gr.Interface(
    fn=detect_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Detection Result"),
    title="Video Detection",
)

demo = gr.TabbedInterface(
    [image_interface, video_interface],
    ["Image", "Video"],
    title="YOLO11 Object Detection",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
