import React from 'react';

interface ContextMenuProps {
  visible: boolean;
  x: number;
  y: number;
  regionId: number | null;
  onTranscribe: () => void;
  onEditVertices: () => void;
  close: () => void;
}

function MenuItem({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: '8px 16px',
        cursor: 'pointer',
        borderBottom: '1px solid #f0f0f0',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f5f5f5';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'white';
      }}
    >
      {children}
    </div>
  );
}

export const ContextMenu: React.FC<ContextMenuProps> = ({
  visible,
  x,
  y,
  onTranscribe,
  onEditVertices,
  close,
}) => {
  if (!visible) return null;

  return (
    <>
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 9999,
        }}
        onClick={close}
      />

      <div
        style={{
          position: 'fixed',
          left: x,
          top: y,
          background: 'white',
          border: '1px solid #d9d9d9',
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          zIndex: 10000,
          minWidth: '150px',
        }}
      >
        <MenuItem onClick={onTranscribe}>🔤 Transcribe Region</MenuItem>
        <MenuItem onClick={onEditVertices}>✏️ Edit Vertices</MenuItem>
      </div>
    </>
  );
};