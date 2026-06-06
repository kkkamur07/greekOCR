"use client";

import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { importPage } from "@/lib/api";

export default function PageImport() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(
    async (image: File, transcription?: File) => {
      setUploading(true);
      setError(null);
      try {
        const summary = await importPage(image, transcription);
        router.push(`/pages/${encodeURIComponent(summary.stem)}`);
        router.refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Import failed");
      } finally {
        setUploading(false);
      }
    },
    [router],
  );

  const handleFiles = (files: FileList | File[]) => {
    const list = Array.from(files);
    const image = list.find((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|tiff?)$/i.test(f.name));
    if (!image) {
      setError("Please choose a page image.");
      return;
    }
    const tx = list.find((f) => f.name.endsWith(".txt"));
    void upload(image, tx);
  };

  return (
    <div>
      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`group w-full rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragOver
            ? "border-blue-400 bg-blue-50/80"
            : "border-gray-200 bg-gray-50/50 hover:border-gray-300 hover:bg-gray-50"
        } disabled:opacity-60`}
      >
        <span
          className={`mx-auto flex h-10 w-10 items-center justify-center rounded-full text-xl transition-colors ${
            dragOver ? "bg-blue-100 text-blue-600" : "bg-white text-gray-400 group-hover:text-gray-600"
          }`}
        >
          +
        </span>
        <p className="mt-3 text-sm font-medium text-gray-900">
          {uploading ? "Uploading…" : "Add a page"}
        </p>
        <p className="mt-1 text-xs text-gray-500">Drop an image here, or click to browse</p>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/tiff,.jpg,.jpeg,.png,.webp,.tif,.tiff,.txt,text/plain"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {error && <p className="mt-2 text-center text-xs text-red-600">{error}</p>}
    </div>
  );
}
