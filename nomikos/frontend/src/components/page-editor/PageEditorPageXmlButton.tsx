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
 * Downloads the current part's PAGE XML export. Unlike the transcription PDF
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
      const blob = await api.getPageXml(projectId, documentId, partId);
      const url = URL.createObjectURL(blob);
      const anchor = globalThis.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadFilename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Failed to download PAGE XML";
      toast.error(message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <button
      type="button"
      className="pe-tb-btn"
      aria-label="Download PAGE XML"
      title="Download PAGE XML"
      disabled={downloading}
      onClick={() => void handleDownload()}
    >
      {downloading ? "XML…" : "XML"}
    </button>
  );
}
