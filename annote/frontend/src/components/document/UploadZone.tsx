import { useRef } from 'react';

type UploadZoneProps = {
  onUpload: (file: File) => void | Promise<void>;
  disabled?: boolean;
  loading?: boolean;
};

export function UploadZone({ onUpload, disabled = false, loading = false }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    await onUpload(file);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="visually-hidden"
        disabled={disabled || loading}
        onChange={(e) => void handleFile(e.target.files?.[0])}
        aria-hidden="true"
        tabIndex={-1}
      />
      <button
        type="button"
        className="upload-zone"
        disabled={disabled || loading}
        aria-label="Upload page images, JPEG, PNG, TIFF, max 50 MB"
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
        <p>{loading ? 'Uploading…' : 'Upload page images'}</p>
        <p className="hint">JPEG, PNG, TIFF, max 50 MB</p>
      </button>
    </>
  );
}
