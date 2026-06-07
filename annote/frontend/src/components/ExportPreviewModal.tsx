"use client";

import { useEffect } from "react";
import { segmentPreviewUrl } from "@/lib/api";

interface ExportPreviewModalProps {
  stem: string;
  segmentId: string;
  segmentNumber: number;
  onClose: () => void;
}

export default function ExportPreviewModal({
  stem,
  segmentId,
  segmentNumber,
  onClose,
}: ExportPreviewModalProps) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const previewUrl = `${segmentPreviewUrl(stem, segmentId)}?t=${Date.now()}`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="max-h-[90vh] w-full max-w-2xl overflow-hidden rounded-xl bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-preview-title"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
          <h2 id="export-preview-title" className="text-sm font-medium text-gray-900">
            Export preview — segment {segmentNumber}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded px-2 py-1 text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
          >
            Close
          </button>
        </div>
        <div className="max-h-[calc(90vh-3.5rem)] overflow-auto bg-gray-100 p-4">
          <p className="mb-3 text-xs text-gray-500">
            Rectified crop as it will appear in export. Check skew, clipping, and margins before exporting
            the page.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt={`Export preview for segment ${segmentNumber}`}
            className="mx-auto max-w-full rounded border border-gray-200 bg-white shadow-sm"
          />
        </div>
      </div>
    </div>
  );
}
