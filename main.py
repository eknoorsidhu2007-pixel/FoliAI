from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

MODEL_PATH = Path(__file__).parent / "model" / "best_model.pth"

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[nn.Module] = None


def load_model() -> nn.Module:
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 15)
    state = torch.load(MODEL_PATH, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


@app.on_event("startup")
def startup() -> None:
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return {
        "class_name": CLASS_NAMES[predicted_idx.item()],
        "confidence": confidence.item(),
    }
