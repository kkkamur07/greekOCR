import { useState } from "react";
import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";

type PageEditorPageXmlButtonProps = {
  projectId: string;
  documentId: string;
  partId: string;
  downloadFilename: string;
};

/**
 * Downloads the current part's PAGE XML bundle: a zip holding the XML next to
 * the full-resolution page image it describes, so the export opens as one unit
 * in Transkribus, eScriptorium and the like. Unlike the transcription PDF
 * there is nothing to preview, so this is a plain download button rather than
 * a pane toggle.
 */
export function PageEditorPageXmlButton({
  projectId,
  documentId,
  partId,
  downloadFilename,
}: PageEditorPageXmlButtonProps) {
  const [downloading, setDownloading] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    try {
      const blob = await api.getPageXmlBundle(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to download PAGE XML and page image";
      toast.error(message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <button
      type="button"
      className="pe-tb-btn"
      aria-label="Download PAGE XML with page image"
      title="Download PAGE XML with the page image (zip)"
      disabled={downloading}
      onClick={() => void handleDownload()}
    >
      {downloading ? "XML…" : "XML"}
    </button>
  );
}
