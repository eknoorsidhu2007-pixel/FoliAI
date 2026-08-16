"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Prediction = {
  class_name: string | null;
  confidence: number;
  matched: boolean;
  message: string | null;
};

const PREDICT_URL = "http://localhost:8000/predict";
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];

function formatClassName(name: string) {
  return name.replace(/_+/g, " ").replace(/\s+/g, " ").trim();
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

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const uploadImage = useCallback(async (file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError("Please upload a JPEG, PNG, WebP, or GIF image.");
      setPrediction(null);
      return;
    }

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextPreview = URL.createObjectURL(file);
    previewUrlRef.current = nextPreview;
    setPreviewUrl(nextPreview);
    setFileName(file.name);
    setError(null);
    setPrediction(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(PREDICT_URL, {
        method: "POST",
        body: formData,
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
            : `Request failed (${response.status})`;
        throw new Error(detail);
      }

      if (
        !payload ||
        typeof payload !== "object" ||
        typeof (payload as Prediction).confidence !== "number" ||
        typeof (payload as Prediction).matched !== "boolean" ||
        !(
          (payload as Prediction).class_name === null ||
          typeof (payload as Prediction).class_name === "string"
        ) ||
        !(
          (payload as Prediction).message === null ||
          typeof (payload as Prediction).message === "string"
        )
      ) {
        throw new Error("Unexpected response from the prediction server.");
      }

      setPrediction(payload as Prediction);
    } catch (err) {
      const message =
        err instanceof TypeError
          ? "Could not reach the prediction server. Is it running on port 8000?"
          : err instanceof Error
            ? err.message
            : "Something went wrong while classifying the image.";
      setError(message);
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const onFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) void uploadImage(file);
    },
    [uploadImage],
  );

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-6">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(true);
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
          onFiles(e.dataTransfer.files);
        }}
        className={`relative flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          isDragging
            ? "border-emerald-500 bg-emerald-50"
            : "border-zinc-300 bg-zinc-50 hover:border-zinc-400 hover:bg-zinc-100"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="sr-only"
          onChange={(e) => {
            onFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Selected leaf preview"
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
            {isDragging ? "Drop image to classify" : "Drag & drop a leaf image"}
          </p>
          <p className="mt-1 text-sm text-zinc-500">
            or click to browse · JPEG, PNG, WebP, GIF
          </p>
          {fileName && (
            <p className="mt-2 truncate text-xs text-zinc-400">{fileName}</p>
          )}
        </div>
      </div>

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

      {prediction && !isLoading && prediction.matched && prediction.class_name && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-5 py-4">
          <p className="text-xs font-medium uppercase tracking-wide text-emerald-700">
            Prediction
          </p>
          <p className="mt-1 text-lg font-semibold text-zinc-900">
            {formatClassName(prediction.class_name)}
          </p>
          <p className="mt-2 text-sm text-zinc-600">
            Confidence:{" "}
            <span className="font-medium text-zinc-900">
              {(prediction.confidence * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      )}

      {prediction && !isLoading && !prediction.matched && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-4">
          <p className="text-xs font-medium uppercase tracking-wide text-amber-700">
            No clear match
          </p>
          <p className="mt-1 text-base font-medium text-zinc-900">
            {prediction.message ??
              "Image does not clearly match any known plant disease class"}
          </p>
          <p className="mt-2 text-sm text-zinc-600">
            Top-class confidence was{" "}
            <span className="font-medium text-zinc-900">
              {(prediction.confidence * 100).toFixed(1)}%
            </span>
            , below the 60% threshold.
          </p>
        </div>
      )}
    </div>
  );
}
