"use client";

interface ZoomControlsProps {
  zoomLevel: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitPage: () => void;
}

export default function ZoomControls({ zoomLevel, onZoomIn, onZoomOut, onFitPage }: ZoomControlsProps) {
  return (
    <div className="image-canvas-zoom flex items-center gap-1 rounded-lg border border-gray-200 bg-white/95 p-1 shadow-sm">
      <button
        type="button"
        onClick={onZoomOut}
        className="flex h-8 w-8 items-center justify-center rounded text-lg hover:bg-gray-50"
        title="Zoom out"
      >
        −
      </button>
      <span className="min-w-[3rem] text-center text-xs text-gray-600">{Math.round(zoomLevel * 100)}%</span>
      <button
        type="button"
        onClick={onZoomIn}
        className="flex h-8 w-8 items-center justify-center rounded text-lg hover:bg-gray-50"
        title="Zoom in"
      >
        +
      </button>
      <button
        type="button"
        onClick={onFitPage}
        className="rounded px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
        title="Fit page"
      >
        Fit
      </button>
    </div>
  );
}
