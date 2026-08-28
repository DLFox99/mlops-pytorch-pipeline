import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

"""
Serves the trained CIFAR-10 classifier over HTTP. Two routes: /health for
k8s probes, /predict for actual inference. Model loads once at startup
from a checkpoint file, not per-request, obviously.
"""

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

app = FastAPI(title="CIFAR-10 Classifier Service")

_state: dict = {"model": None, "device": None}

_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]
)


def _checkpoint_path() -> Path:
    env_path = os.environ.get("MODEL_CHECKPOINT_PATH")
    if env_path:
        return Path(env_path)
    return Path("/app/checkpoints/classifier_v1.pt")


@app.on_event("startup")
def load_model() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _checkpoint_path()

    if not checkpoint_path.exists():
        _state["model"] = None
        _state["device"] = device
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(num_classes=checkpoint.get("num_classes", 10))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device


@app.get("/health")
def health() -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    tensor = _transform(img).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_idx = int(torch.argmax(logits, dim=1).item())

    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {
            CIFAR10_CLASSES[i]: round(p, 4) for i, p in enumerate(probabilities)
        },
    }
