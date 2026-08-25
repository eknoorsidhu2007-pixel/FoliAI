"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type PredictionItem = {
  class_name: string;
  confidence: number;
};

type Prediction = {
  class_name: string | null;
  confidence: number;
  matched: boolean;
  message: string | null;
  predictions: PredictionItem[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";
const PREDICT_URL = `${API_BASE}/predict`;

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10MB (matches backend)
const MAX_IMAGE_DIMENSION = 800;
const REQUEST_TIMEOUT_MS = 30_000;

function formatClassName(name: string) {
  return name.replace(/_+/g, " ").replace(/\s+/g, " ").trim();
}

/** §6 Resize/compress on canvas; longest side capped at MAX_IMAGE_DIMENSION. */
async function resizeImageForUpload(file: File): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(
      1,
      MAX_IMAGE_DIMENSION / Math.max(bitmap.width, bitmap.height),
    );
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return file;
    }
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.85),
    );
    if (!blob) return file;

    const baseName = file.name.replace(/\.[^.]+$/, "") || "upload";
    return new File([blob], `${baseName}.jpg`, { type: "image/jpeg" });
  } catch {
    // Fall back to the original file if canvas/bitmap resize fails.
    return file;
  }
}

function isPredictionPayload(payload: unknown): payload is Prediction {
  if (!payload || typeof payload !== "object") return false;
  const p = payload as Prediction;
  if (typeof p.confidence !== "number" || typeof p.matched !== "boolean") {
    return false;
  }
  if (!(p.class_name === null || typeof p.class_name === "string")) return false;
  if (!(p.message === null || typeof p.message === "string")) return false;
  if (!Array.isArray(p.predictions) || p.predictions.length === 0) return false;
  return p.predictions.every(
    (item) =>
      item &&
      typeof item.class_name === "string" &&
      typeof item.confidence === "number",
  );
}

export default function ImageUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  // §11 Revoke object URLs on unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const setPreviewFromFile = useCallback((file: File) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextPreview = URL.createObjectURL(file);
    previewUrlRef.current = nextPreview;
    setPreviewUrl(nextPreview);
  }, []);

  const resetUpload = useCallback(() => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
    setFileName(null);
    setPrediction(null);
    setError(null);
    setIsDragging(false);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const openFilePicker = useCallback(() => {
    if (isLoading) return;
    inputRef.current?.click();
  }, [isLoading]);

  const uploadImage = useCallback(
    async (file: File) => {
      if (isLoading) return;

      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Please upload a JPEG, PNG, WebP, or GIF image.");
        setPrediction(null);
        return;
      }

      if (file.size > MAX_UPLOAD_BYTES) {
        setError("File too large. Maximum upload size is 10MB.");
        setPrediction(null);
        return;
      }

      setFileName(file.name);
      setError(null);
      setPrediction(null);
      setIsLoading(true);

      // Preview the original selection immediately
      setPreviewFromFile(file);

      const controller = new AbortController();
      const timeoutId = window.setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS,
      );

      try {
        // §6 Compress before upload
        const uploadFile = await resizeImageForUpload(file);
        // Refresh preview to the compressed image used for inference
        setPreviewFromFile(uploadFile);

        const formData = new FormData();
        formData.append("file", uploadFile);

        // §7 Timed fetch
        const response = await fetch(PREDICT_URL, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        let payload: unknown = null;
        try {
          payload = await response.json();
        } catch {
          payload = null;
        }

        if (!response.ok) {
          const detail =
            payload &&
            typeof payload === "object" &&
            "detail" in payload &&
            typeof (payload as { detail: unknown }).detail === "string"
              ? (payload as { detail: string }).detail
              : response.status === 413
                ? "File too large. Maximum upload size is 10MB."
                : `Request failed (${response.status})`;
          throw new Error(detail);
        }

        if (!isPredictionPayload(payload)) {
          throw new Error("Unexpected response from the prediction server.");
        }

        setPrediction(payload);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setError("Request timed out, please try again.");
        } else if (err instanceof TypeError) {
          setError(
            "Could not reach the prediction server. Is it running on port 8000?",
          );
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Something went wrong while classifying the image.");
        }
        setPrediction(null);
      } finally {
        window.clearTimeout(timeoutId);
        setIsLoading(false);
      }
    },
    [isLoading, setPreviewFromFile],
  );

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (isLoading) return;
      const file = files?.[0];
      if (file) void uploadImage(file);
    },
    [isLoading, uploadImage],
  );

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-6">
      {/* §8 Disabled while loading; §12 drag-active styles; §14 keyboard a11y */}
      <div
        role="button"
        tabIndex={isLoading ? -1 : 0}
        aria-label="Upload an image"
        aria-disabled={isLoading}
        onKeyDown={(e) => {
          if (isLoading) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFilePicker();
          }
        }}
        onClick={() => openFilePicker()}
        onDragEnter={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!isLoading) setIsDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!isLoading) setIsDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
          if (!isLoading) onFiles(e.dataTransfer.files);
        }}
        className={`relative flex min-h-56 flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 ${
          isLoading
            ? "cursor-not-allowed border-zinc-200 bg-zinc-100 opacity-70"
            : isDragging
              ? "cursor-pointer border-emerald-500 bg-emerald-50"
              : "cursor-pointer border-zinc-300 bg-zinc-50 hover:border-zinc-400 hover:bg-zinc-100"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="sr-only"
          disabled={isLoading}
          onChange={(e) => {
            onFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt={fileName ? `Preview of uploaded image ${fileName}` : "Uploaded image preview"}
            className="max-h-48 w-auto rounded-lg object-contain"
          />
        ) : (
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-zinc-200">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.75"
              className="h-7 w-7 text-emerald-600"
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
        )}

        <div>
          <p className="text-base font-medium text-zinc-900">
            {isLoading
              ? "Uploading and classifying…"
              : isDragging
                ? "Drop image to classify"
                : "Drag & drop a leaf image"}
          </p>
          <p className="mt-1 text-sm text-zinc-500">
            {isLoading ? "Please wait" : "or click / press Enter to browse"}
          </p>
          {fileName && (
            <p className="mt-2 truncate text-xs text-zinc-400">{fileName}</p>
          )}
        </div>
      </div>

      {/* §13 Accepted types + size limit helper text */}
      <p className="text-center text-sm text-zinc-500">
        Accepted: JPEG, PNG, WebP, GIF · Max size: 10MB · Images are resized to
        800px before upload
      </p>

      {isLoading && (
        <div
          className="flex items-center justify-center gap-3 rounded-xl bg-zinc-100 px-4 py-5 text-sm text-zinc-700"
          role="status"
          aria-live="polite"
        >
          <span
            className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"
            aria-hidden
          />
          Classifying image…
        </div>
      )}

      {error && !isLoading && (
        <div
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* §9 Top-3 ranked list with confidence bars */}
      {prediction && !isLoading && (
        <div
          className={`rounded-xl border px-5 py-4 ${
            prediction.matched
              ? "border-emerald-200 bg-emerald-50"
              : "border-amber-200 bg-amber-50"
          }`}
        >
          <p
            className={`text-xs font-medium uppercase tracking-wide ${
              prediction.matched ? "text-emerald-700" : "text-amber-700"
            }`}
          >
            {prediction.matched ? "Top predictions" : "No clear match"}
          </p>

          {!prediction.matched && (
            <p className="mt-1 text-base font-medium text-zinc-900">
              {prediction.message ??
                "Image does not clearly match any known plant disease class"}
            </p>
          )}

          <ol className="mt-4 flex flex-col gap-3" aria-label="Top 3 predictions">
            {prediction.predictions.map((item, index) => {
              const pct = Math.max(0, Math.min(100, item.confidence * 100));
              return (
                <li key={`${item.class_name}-${index}`}>
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="font-medium text-zinc-900">
                      <span className="mr-2 text-zinc-400">{index + 1}.</span>
                      {formatClassName(item.class_name)}
                    </span>
                    <span className="shrink-0 tabular-nums text-zinc-600">
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                  <div
                    className="mt-1.5 h-2 overflow-hidden rounded-full bg-white/80 ring-1 ring-zinc-200/80"
                    role="progressbar"
                    aria-valuenow={Math.round(pct)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${formatClassName(item.class_name)} confidence`}
                  >
                    <div
                      className={`h-full rounded-full transition-[width] ${
                        prediction.matched ? "bg-emerald-600" : "bg-amber-500"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ol>

          {!prediction.matched && (
            <p className="mt-3 text-sm text-zinc-600">
              Top-class confidence was below the 60% threshold.
            </p>
          )}

          {/* §10 Reset */}
          <button
            type="button"
            onClick={resetUpload}
            className="mt-5 w-full rounded-lg border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
          >
            Upload another image
          </button>
        </div>
      )}

      {error && !isLoading && !prediction && (
        <button
          type="button"
          onClick={resetUpload}
          className="rounded-lg border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
        >
          Upload another image
        </button>
      )}
    </div>
  );
}
