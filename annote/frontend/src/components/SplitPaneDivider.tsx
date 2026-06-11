"use client";

interface SplitPaneDividerProps {
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerUp: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerCancel: (e: React.PointerEvent<HTMLDivElement>) => void;
  onDoubleClick: () => void;
}

export default function SplitPaneDivider({
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerCancel,
  onDoubleClick,
}: SplitPaneDividerProps) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize panels"
      title="Drag to resize · double-click to reset"
      className="group relative z-10 w-1.5 shrink-0 cursor-col-resize touch-none bg-gray-200 transition-colors hover:bg-indigo-300 active:bg-indigo-400"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onDoubleClick={onDoubleClick}
    >
      <div className="absolute inset-y-0 -left-1 -right-1" />
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-8 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gray-400 opacity-0 transition-opacity group-hover:opacity-100" />
    </div>
  );
}
