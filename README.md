# FoliAI


# 🌿 FoliAI

A plant disease classifier: upload a photo of a leaf, get back the most likely disease (or a clean bill of health) with a confidence score — backed by a fine-tuned ResNet-18 model served over FastAPI, with a Next.js frontend.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-frontend-black.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## What it does

- Classifies leaf images into **15 classes** across pepper, potato, and tomato plants — bacterial spot, early/late blight, leaf mold, spider mites, mosaic virus, and healthy, among others
- Returns the **top-3 predictions** with confidence scores, not just a single label
- **Rejects low-confidence predictions** instead of forcing a guess — if the top prediction is below a 60% confidence threshold, the API reports "no clear match" rather than a false positive

## Demo

> _Add a screenshot or GIF of the upload → prediction flow here._

## How it works

### Backend (`main.py`, FastAPI)

- ResNet-18 backbone with a replaced final layer for the 15-class head, loaded from a local checkpoint (`model/best_model.pth`)
- **Temperature-scaled softmax** (T = 2.5) before the confidence threshold — cross-entropy-trained nets are often overconfident on out-of-distribution inputs (a blank desk, a solid color), so softening the distribution before gating on 60% helps catch those cases instead of confidently misclassifying them
- **Input validation**: images are decoded and re-verified as real image data (not just trusted by file extension or `Content-Type` header), uploads are capped at 10MB, and unhandled errors return a generic message instead of leaking a Python traceback
- **CORS** locked to the frontend's origin, not a wildcard

### Frontend (`frontend/`, Next.js + React)

- Drag-and-drop or click-to-upload, with keyboard accessibility
- Client-side image resize (longest side capped at 800px, re-encoded as JPEG) before upload, to keep requests small and fast
- Displays the top-3 predictions with confidence bars, and a distinct "no clear match" state when the model isn't confident

## Tech stack

| Layer | Tools |
|---|---|
| ML | PyTorch, torchvision (ResNet-18) |
| Backend | FastAPI, Pillow, python-multipart |
| Frontend | Next.js, React, TypeScript, Tailwind CSS |

## Getting started

### Prerequisites

- Python 3.9+
- Node.js 18+
- A trained model checkpoint at `model/best_model.pth`

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend expects the API at `http://localhost:8000` — override with the `NEXT_PUBLIC_API_URL` environment variable.

## API reference

### `POST /predict`

Multipart form upload with a `file` field (JPEG, PNG, WebP, or GIF, max 10MB).

**Response:**

```json
{
  "class_name": "Tomato_Late_blight",
  "confidence": 0.87,
  "matched": true,
  "message": null,
  "predictions": [
    { "class_name": "Tomato_Late_blight", "confidence": 0.87 },
    { "class_name": "Tomato_Early_blight", "confidence": 0.09 },
    { "class_name": "Tomato_healthy", "confidence": 0.02 }
  ]
}
```

### `GET /health`

Reports whether the model is loaded.

```json
{ "status": "ok", "model_loaded": true }
```

## Supported classes

<details>
<summary>Click to expand — 15 classes across 3 crops</summary>

- **Pepper**: bacterial spot, healthy
- **Potato**: early blight, late blight, healthy
- **Tomato**: bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, mosaic virus, yellow leaf curl virus, healthy

</details>

## Roadmap

- [ ] Model soup / weight averaging across multiple fine-tuning runs for improved accuracy and robustness
- [ ] Grad-CAM visualization to verify predictions are grounded in actual lesion features, not background artifacts
- [ ] Expand class coverage to additional crops

## License

MIT — see [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests welcome. For major changes, please open an issue first to discuss what you'd like to change.
