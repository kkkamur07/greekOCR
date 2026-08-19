import { useRef } from "react";

type UploadZoneProps = {
  onUpload: (files: File[]) => void | Promise<void>;
  disabled?: boolean;
  loading?: boolean;
  /** Narration for a long batch ("Splitting scan.pdf · page 3/12"). */
  progress?: string | null;
};

export function UploadZone({
  onUpload,
  disabled = false,
  loading = false,
  progress = null,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (files.length === 0) return;
    await onUpload(files);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,application/pdf,.pdf"
        multiple
        className="visually-hidden"
        disabled={disabled || loading}
        onChange={(e) => void handleFiles(e.target.files)}
        aria-hidden="true"
        tabIndex={-1}
      />
      <button
        type="button"
        className="upload-zone"
        disabled={disabled || loading}
        aria-label="Upload page images or PDFs; JPEG, PNG, TIFF, or PDF split into one page per sheet"
        onClick={() => inputRef.current?.click()}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth="1.5"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
          />
        </svg>
        <p>
          {loading ? (progress ?? "Uploading…") : "Upload page images or PDFs"}
        </p>
        <p className="hint">
          Drop files anywhere · JPEG, PNG, TIFF — a PDF becomes one page per
          sheet
        </p>
      </button>
    </>
  );
}
