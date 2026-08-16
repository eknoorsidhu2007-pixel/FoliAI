import io
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

logger = logging.getLogger("foliai")
logging.basicConfig(level=logging.INFO)

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
CONFIDENCE_THRESHOLD = 0.60
# Softmax from CE-trained nets is overconfident on OOD inputs (blank desks,
# solid colors often land at ~90%+). Dividing logits by T > 1 softens the
# distribution so the 60% gate can reject those cases.
SOFTMAX_TEMPERATURE = 2.5
UNKNOWN_MESSAGE = (
    "Image does not clearly match any known plant disease class"
)

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


def _unwrap_state_dict(raw: Any) -> dict:
    """Accept a raw state_dict or common checkpoint wrappers."""
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(raw)!r}")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        nested = raw.get(key)
        if isinstance(nested, dict) and any(
            isinstance(v, torch.Tensor) for v in nested.values()
        ):
            raw = nested
            break

    if any(k.startswith("module.") for k in raw):
        raw = {k.removeprefix("module."): v for k, v in raw.items()}

    return raw


def load_model() -> nn.Module:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))

    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(raw)

    expected = set(net.state_dict().keys())
    loaded = set(state.keys())
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match ResNet18+15-class head: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    fc_weight = state.get("fc.weight")
    if not isinstance(fc_weight, torch.Tensor) or tuple(fc_weight.shape) != (
        len(CLASS_NAMES),
        512,
    ):
        raise RuntimeError(
            f"fc.weight has unexpected shape {getattr(fc_weight, 'shape', None)}; "
            f"expected ({len(CLASS_NAMES)}, 512)"
        )

    net.load_state_dict(state, strict=True)
    net.eval()

    if net.training:
        raise RuntimeError("Model remained in training mode after load_model()")

    bn_training = [
        name
        for name, module in net.named_modules()
        if isinstance(module, nn.BatchNorm2d) and module.training
    ]
    if bn_training:
        raise RuntimeError(f"BatchNorm still in training mode: {bn_training}")

    param_count = sum(p.numel() for p in net.parameters())
    logger.info(
        "Loaded %s (%s params, %d classes). model.training=%s",
        MODEL_PATH,
        f"{param_count:,}",
        len(CLASS_NAMES),
        net.training,
    )
    return net


@app.on_event("startup")
def startup() -> None:
    global model
    model = load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_training": bool(model.training) if model is not None else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Ensure eval mode + frozen BN running stats on every request (no train leftovers).
    model.eval()

    try:
        # Read bytes explicitly so we never reuse a partially-consumed upload stream.
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.load()
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    tensor = preprocess(image).unsqueeze(0)

    # Debug: confirm each upload yields a distinct preprocessed tensor.
    print(
        f"[predict] tensor shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f}"
    )

    with torch.no_grad():
        # Softmax MUST use dim=1 (class axis). dim=0 with batch size 1
        # makes every class probability exactly 1.0 — false 100% confidence.
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected logits shape {tuple(logits.shape)}",
            )

        probabilities = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    conf_value = float(confidence.item())
    class_idx = int(predicted_idx.item())
    print(
        f"[predict] top_class={CLASS_NAMES[class_idx]} "
        f"confidence={conf_value:.6f} "
        f"logit_max={logits.max().item():.6f} "
        f"logit_min={logits.min().item():.6f}"
    )

    if conf_value < CONFIDENCE_THRESHOLD:
        return {
            "class_name": None,
            "confidence": conf_value,
            "matched": False,
            "message": UNKNOWN_MESSAGE,
        }

    return {
        "class_name": CLASS_NAMES[class_idx],
        "confidence": conf_value,
        "matched": True,
        "message": None,
    }
