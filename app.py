# =========================================
# app.py - Lung X-ray Prediction API
# =========================================

import io
import os
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
model = Net().to(device)

artifact_root = "artifacts"

if not os.path.exists(artifact_root):
    raise FileNotFoundError("Artifacts folder not found. Train the model first.")

# Get latest timestamp folder
latest_folder = sorted(os.listdir(artifact_root))[-1]

model_path = os.path.join(
    artifact_root,
    latest_folder,
    "model_training",
    "model.pt"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()

print("Model loaded from:", model_path)

# -----------------------------------------
# 4. Define Image Transform
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
            confidence, predicted_class = torch.max(probabilities, 1)

        prediction_index = predicted_class.item()
        prediction_label = label_map.get(prediction_index, "Unknown")

        return JSONResponse({
            "prediction_index": prediction_index,
            "prediction_label": prediction_label,
            "confidence": float(confidence.item())
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )