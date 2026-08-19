import { useEffect, useRef, useState } from "react";

/**
 * Make the whole window a file drop target.
 *
 * Listening on `window` rather than an element means the researcher can let
 * go anywhere on the page - the drop zone button stays as the click-to-pick
 * alternative, not the only landing strip. Returns whether a file drag is
 * currently over the window, which is what shows the "drop to upload"
 * overlay.
 *
 * Drags that carry no files (text selections, the part list's own future
 * drags) are ignored entirely: no overlay, no preventDefault, so the
 * browser's default behaviour for them is untouched.
 */
export function useFileDrop(
  onFiles: (files: File[]) => void,
  enabled: boolean,
): boolean {
  const [dragActive, setDragActive] = useState(false);
  // Held in a ref so a caller may pass an inline closure without re-binding
  // the window listeners on every render.
  const onFilesRef = useRef(onFiles);
  onFilesRef.current = onFiles;
  // dragenter/dragleave fire for every nested element the drag crosses; only
  // the outermost balance says whether the pointer actually left the window.
  const depthRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      depthRef.current = 0;
      setDragActive(false);
      return;
    }

    const carriesFiles = (event: DragEvent) =>
      !!event.dataTransfer &&
      Array.from(event.dataTransfer.types).includes("Files");

    const handleEnter = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depthRef.current += 1;
      setDragActive(true);
    };
    const handleOver = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      // Without this the browser navigates to the dropped file.
      event.preventDefault();
    };
    const handleLeave = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depthRef.current = Math.max(0, depthRef.current - 1);
      if (depthRef.current === 0) setDragActive(false);
    };
    const handleDrop = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depthRef.current = 0;
      setDragActive(false);
      const files = Array.from(event.dataTransfer?.files ?? []);
      if (files.length > 0) onFilesRef.current(files);
    };

    window.addEventListener("dragenter", handleEnter);
    window.addEventListener("dragover", handleOver);
    window.addEventListener("dragleave", handleLeave);
    window.addEventListener("drop", handleDrop);
    return () => {
      window.removeEventListener("dragenter", handleEnter);
      window.removeEventListener("dragover", handleOver);
      window.removeEventListener("dragleave", handleLeave);
      window.removeEventListener("drop", handleDrop);
    };
  }, [enabled]);

  return dragActive;
}
