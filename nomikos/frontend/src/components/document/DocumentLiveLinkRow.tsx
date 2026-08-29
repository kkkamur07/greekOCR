import Link from "next/link";
import {
  publicDocumentPath,
  publicDocumentUrl,
} from "../../utils/publicDocumentUrl";
import { toast } from "../ui/toast";

type DocumentLiveLinkRowProps = {
  projectId: string;
  documentId: string;
  /**
   * Owner-only on the wire. A collaborator's copy of a published document
   * carries null here, and the row says so rather than offering a link that
   * would 404 for whoever opened it.
   */
  publicShareToken: string | null;
};

/**
 * The live URL of a published document, shown while it is published.
 *
 * It sits under the action row rather than inside the Publish menu because the
 * fact that a document is readable by anyone holding a link is a standing
 * condition, not a menu item: it should be visible without opening anything.
 */
export function DocumentLiveLinkRow({
  projectId,
  documentId,
  publicShareToken,
}: DocumentLiveLinkRowProps) {
  if (!publicShareToken) {
    return (
      <p className="doc-live-link doc-live-link--muted">
        This document is live. Only the project owner can get the secret link.
      </p>
    );
  }

  const path = publicDocumentPath(projectId, documentId, publicShareToken);
  const url = publicDocumentUrl(projectId, documentId, publicShareToken);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(url);
      toast.success("Secret link copied");
    } catch {
      toast.error("Could not copy the link");
    }
  }

  return (
    <div className="doc-live-link">
      <span className="doc-live-link__label">Live at</span>
      <input
        className="doc-live-link__url"
        type="text"
        readOnly
        value={url}
        aria-label="Public document link"
      />
      <Link
        href={path}
        className="btn btn-ghost btn-xs"
        target="_blank"
        rel="noopener noreferrer"
      >
        Open
      </Link>
      <button
        type="button"
        className="btn btn-outline btn-xs"
        onClick={() => void handleCopy()}
      >
        Copy
      </button>
    </div>
  );
}
