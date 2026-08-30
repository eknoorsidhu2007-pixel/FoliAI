# FoliAI

FoliAI
A plant disease classifier: upload a photo of a leaf, get back the most likely disease (or "healthy") with a confidence score, backed by a fine-tuned ResNet-18 model served over a FastAPI backend and a Next.js frontend.
What it does

Classifies leaf images into 15 classes across pepper, potato, and tomato plants (bacterial spot, early/late blight, leaf mold, spider mites, mosaic virus, and healthy, among others)
Returns the top-3 predictions with confidence scores, not just a single label
Rejects low-confidence predictions instead of forcing a guess — if the top prediction is below a 60% confidence threshold, the API reports "no clear match" rather than a false positive
How it works
Backend (main.py, FastAPI)

ResNet-18 backbone with a replaced final layer for the 15-class head, loaded from a local checkpoint (model/best_model.pth)
Temperature-scaled softmax (T = 2.5) before the confidence threshold — cross-entropy-trained nets are often overconfident on out-of-distribution inputs (a blank desk, a solid color), so softening the distribution before gating on 60% helps catch those cases instead of confidently misclassifying them
Input validation: images are decoded and re-verified as real image data (not just trusted by file extension or Content-Type header), uploads are capped at 10MB, and unhandled errors return a generic message instead of leaking a Python traceback
CORS locked to the frontend's origin, not a wildcard
Frontend (frontend/, Next.js + React)

Drag-and-drop or click-to-upload, with keyboard accessibility
Client-side image resize (longest side capped at 800px, re-encoded as JPEG) before upload, to keep requests small and fast
Displays the top-3 predictions with confidence bars, and a distinct "no clear match" state when the model isn't confident
Tech stack

ML: PyTorch, torchvision (ResNet-18)
Backend: FastAPI, Pillow, python-multipart
Frontend: Next.js, React, TypeScript, Tailwind CSS
Running locally
Backend

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
Frontend

cd frontend
npm install
npm run dev
By default the frontend expects the API at http://localhost:8000 (override with NEXT_PUBLIC_API_URL).
API
POST /predict — multipart form upload with a file field (JPEG, PNG, WebP, or GIF, max 10MB). Returns:

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
GET /health — reports whether the model is loaded.
