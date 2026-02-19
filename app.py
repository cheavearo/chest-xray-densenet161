# Import libraries
import time
import torch
import numpy as np

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time
import cv2
import torch
import gradio as gr

import onnxruntime
from PIL import Image, ImageDraw, ImageFont
from src.ui_display.ui_display import format_result_html
from src.database.mongo_db import MongoDBNetwork

css = """
.report {
    font-family: Arial, sans-serif;
    padding: 15px;
    background: white;
    border-radius: 10px;
}

.diagnosis {
    margin-bottom: 20px;
}

.diag-title {
    font-size: 13px;
    color: #6b7280;
}

.diag-main {
    font-size: 22px;
    font-weight: 600;
    color: #1f2937;
}

.diag-sub {
    font-size: 13px;
    color: #374151;
    margin-top: 2px;
}

.row {
    margin-bottom: 14px;
}

.row-header {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    margin-bottom: 4px;
}

.label {
    color: #374151;
}

.percent {
    color: #1d4ed8;
    font-weight: 500;
}

.progress {
    width: 100%;
    height: 10px;
    background-color: #dbeafe;   /* light blue background */
    border-radius: 6px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: #2563eb;   /* stronger blue */
    border-radius: 6px;
}
"""

theme = gr.themes.Default(primary_hue="blue").set(

    button_primary_background_fill_hover="*primary_400",
)

#theme = gr.themes.Ocean(primary_hue="violet", secondary_hue="violet")

# Load ONNX model
ort_session = onnxruntime.InferenceSession("final_model/best_full_finetune_4_classes_model.onnx", providers=["CPUExecutionProvider"])

# Initialize MongDB 
mong_db_obj = MongoDBNetwork()

# Class names
class_names = {
    0: "covid",
    1: "lung_opacity",
    2: "normal",
    3: "viral_pnuemonia"
}
# Inference transform (single image)
transform = A.Compose([
    A.Resize(224, 224),  # Match input size of DenseNet
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()          # Converts to torch tensor, shape [C,H,W]
])

device='cpu' 

def infer_onnx(patient_id, image_path, top_k=5):
    """
    Perform image classification inference using ONNX Runtime.

    Args:
        image_path (str): Path to input image.
        ort_session (onnxruntime.InferenceSession): Loaded ONNX model session.
        transform (dict): Preprocessing transforms (Albumentations or torchvision).
        class_names (list): List mapping class indices to names.
        device (str): 'cpu' or 'cuda'
        top_k (int): Number of top predictions to return.

    Returns:
        List of tuples: [(class_name, probability), ...]
    """
    start_time = time.time()
    # Step 1: Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    output_img = image.copy()
    image = np.array(image)  # Albumentations expects NumPy

    # Apply Albumentations transform
    augmented = transform(image=image)
    img_tensor = augmented["image"]  # Tensor shape [C, H, W]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    
    # Convert to numpy float32 for ONNX Runtime
    onnx_input = img_tensor.cpu().numpy().astype(np.float32)

    # Step 2: Prepare input dict for ONNX
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: onnx_input}

    # Step 3: Run inference
    ort_outputs = ort_session.run(None, ort_inputs)[0]  # shape [1, num_classes]
    
    # Step 4: Compute softmax
    probs = torch.nn.functional.softmax(torch.from_numpy(ort_outputs[0]), dim=0)

    # Step 5: Get top-K
    top_k = min(top_k, probs.shape[0])
    top_probs, top_idxs = probs.topk(top_k)
    results = [(class_names[idx.item()], prob.item()) for idx, prob in zip(top_idxs, top_probs)]
    top_label, top_conf = results[0]
    top_conf_percentage = top_conf * 100
    text = f"{patient_id}_{top_label} : {top_conf_percentage:.2f}%"
    width, height = output_img.size
    if width and height < 200:
        font_size = int(width * 0.15)
    if 200<= width and height <=600:
        font_size = int(width * 0.07)
    else:
        font_size = int(width * 0.05)
    print(f"Image Size (W,H) : ({width}, {height})")

    # Annotate the label to the ouput image
    draw = ImageDraw.Draw(output_img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((0.05*width, 0.05*height), text=text, font=font, fill =(255, 0, 0))
    # Save the output image
    safe_text = text.replace(":", "_").replace(" ", "_").replace("%", "")
    output_path = os.path.join(os.getcwd(), f"output_image_{safe_text}.jpg")
    output_img.save(output_path)
    #html = format_result_html(results)
    outputs = {
        "image" : output_path,
        "results" : results,
        "patient_id" : patient_id,
        "prediction" : top_label,
        "confidence" : float(top_conf_percentage),
    }
    inferecing_time = time.time() - start_time
    print(f"⏳ Total Time: {inferecing_time:.2f} seconds\n")
    
    return outputs

def render_ui(outputs):
    html = format_result_html(outputs["results"])
    return outputs["image"], html

def save_inferenced_image_to_mongodb(outputs):
    mong_db_obj = MongoDBNetwork()
    mong_db_obj.save_inferenced_image(
        outputs["image"],
        outputs["patient_id"],
        outputs["prediction"],
        outputs["confidence"]   # fix key name
    )

def run_pipeline(patient_id, image_path):
    if not patient_id:
        raise gr.Error("Please enter Patient ID")

    if image_path is None:
        raise gr.Error("Please upload an X-ray image")

    outputs = infer_onnx(patient_id, image_path)
    image, html = render_ui(outputs)

    return image, html, outputs  # exactly 3 outputs for Gradio

def save_result(outputs):
    
    if outputs is None:
        raise gr.Error("Run prediction first")

    save_inferenced_image_to_mongodb(outputs)
    return gr.Info("Saved to database ℹ️")


with gr.Blocks() as app:
    
    # Browser tab title and visible header
    gr.HTML("""
    <div style="text-align:center; margin-bottom:20px;">
        <h1 style="font-size:64px; font-weight:bold; color:#0F172A; margin:0;">
            Chest X-Ray Prediction System
        </h1>
        <p style="font-size:20px; color:#555;">
        This AI-powered system analyzes chest X-ray images to classify lung conditions into four categories:<br>
        <b>COVID-19, Lung Opacity, Normal, Viral Pneumonia</b>
        </p>

    </div>
""")

    # Input row
    with gr.Row():
        with gr.Column(scale=1):
            input_id = gr.Textbox(type="text", label="Patient ID")
            input_image = gr.Image(type="filepath", label="Upload X-Ray Image")
            

        with gr.Column(scale=1):
            output_html = gr.HTML(label="Prediction")
            output_image = gr.Image(label="Processed X-ray", type="filepath")

    with gr.Row():
        prediction_stage = gr.State(None)

        clear_btn = gr.Button("Clear")
        predict_btn = gr.Button("Predict", variant='huggingface')
        save_btn = gr.Button("Save Result to Database", variant=" primary")
        predict_btn.click(
            fn=run_pipeline,
            inputs=[input_id, input_image],
            outputs=[output_image, output_html, prediction_stage] # prediction stage must be the last output.
        )
        save_btn.click(
            fn=save_result,
            inputs=prediction_stage,
            outputs= [],

        )
        clear_btn.click(
            fn=lambda: (None, None, None, None),
            inputs=[],
            outputs=[input_image, output_image, output_html]
        )
       
if __name__=="__main__":
    app.launch(css=css, theme=theme)







