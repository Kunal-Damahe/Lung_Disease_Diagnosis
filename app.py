# =========================================
# app.py - Lung X-ray Prediction API
# =========================================

import io
import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from xray.ml.model.arch import Net  

# -----------------------------------------
# 1. Initialize FastAPI app
# -----------------------------------------
app = FastAPI(title="Lung Disease Diagnosis API")

# -----------------------------------------
# 2. Device configuration
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------
# 3. Load Latest Trained Model Automatically
# -----------------------------------------
model = Net(pretrained=False).to(device)

artifact_root = "artifacts"

configured_model_path = os.getenv("MODEL_PATH")
if configured_model_path:
    model_path = Path(configured_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Configured model not found: {model_path}")
else:
    # Select the newest successful checkpoint, ignoring incomplete artifact runs.
    model_candidates = list(Path(artifact_root).glob("*/model_training/model.pt"))
    if not model_candidates:
        raise FileNotFoundError("No trained model checkpoint found. Train the model first.")
    model_path = max(model_candidates, key=lambda path: path.stat().st_mtime)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
model.load_state_dict(state_dict)

model.eval()

print("Model loaded from:", model_path)

# -----------------------------------------
# 4. Define Image Transform
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# 5. Label Mapping
# -----------------------------------------
label_map = {
    0: "Normal",
    1: "Pneumonia"
}

# -----------------------------------------
# 6. Health Check Route
# -----------------------------------------
@app.get("/")
def home():
    return {"message": "Lung Disease Diagnosis API is running"}

# -----------------------------------------
# 7. Prediction Route
# -----------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pneumonia_probability = probabilities[:, 1]
            predicted_class = (pneumonia_probability >= decision_threshold).long()
            decision_margin = torch.abs(pneumonia_probability - decision_threshold)

        prediction_index = predicted_class.item()
        prediction_label = label_map.get(prediction_index, "Unknown")

        return JSONResponse({
            "prediction_index": prediction_index,
            "prediction_label": prediction_label,
            "pneumonia_probability": float(pneumonia_probability.item()),
            "decision_threshold": decision_threshold,
            "decision_margin": float(decision_margin.item()),
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
