import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react';
import type { Region } from '../../types';
import { PublicZoomSurface } from './PublicZoomSurface';

type PublicPageCanvasProps = {
  imageUrl: string;
  layoutWidth: number;
  layoutHeight: number;
  regions: Region[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
};

type Size = { width: number; height: number };

export function PublicPageCanvas({
  imageUrl,
  layoutWidth,
  layoutHeight,
  regions,
  selectedRegionId,
  onSelectRegion,
}: PublicPageCanvasProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [displaySize, setDisplaySize] = useState<Size | null>(null);
  const [coordSize, setCoordSize] = useState<Size>({
    width: layoutWidth,
    height: layoutHeight,
  });

  const syncDisplaySize = useCallback(() => {
    const image = imageRef.current;
    if (!image) return;
    const width = image.clientWidth;
    const height = image.clientHeight;
    if (width > 0 && height > 0) {
      setDisplaySize({ width, height });
    }
  }, []);

  useEffect(() => {
    const image = imageRef.current;
    if (!image || typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(() => {
      syncDisplaySize();
    });
    observer.observe(image);
    return () => observer.disconnect();
  }, [syncDisplaySize, imageUrl]);

  const handleImageLoad = () => {
    const image = imageRef.current;
    if (!image) return;
    const { naturalWidth, naturalHeight } = image;
    if (naturalWidth > 0 && naturalHeight > 0) {
      setCoordSize({ width: naturalWidth, height: naturalHeight });
    } else if (layoutWidth > 0 && layoutHeight > 0) {
      setCoordSize({ width: layoutWidth, height: layoutHeight });
    }
    syncDisplaySize();
  };
  const selectWithKeyboard = (
    event: KeyboardEvent<SVGPolygonElement>,
    regionId: number,
    selected: boolean,
  ) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    onSelectRegion(selected ? null : regionId);
  };

  return (
    <PublicZoomSurface ariaLabel="Manuscript page viewer">
      <div className="public-page-canvas__frame">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Manuscript page"
          draggable={false}
          onLoad={handleImageLoad}
          className="public-page-canvas__image"
        />
        {displaySize && coordSize.width > 0 && coordSize.height > 0 && (
          <svg
            className="public-page-canvas__overlay"
            viewBox={`0 0 ${coordSize.width} ${coordSize.height}`}
            preserveAspectRatio="none"
            style={{
              width: displaySize.width,
              height: displaySize.height,
            }}
            aria-hidden={regions.length === 0}
            role="group"
            aria-label="Selectable transcription lines"
          >
            {regions.map((region) => {
              const isSelected = region.id === selectedRegionId;
              const points = region.boundary.map(([x, y]) => `${x},${y}`).join(' ');
              return (
                <polygon
                  key={region.id}
                  role="button"
                  tabIndex={0}
                  points={points}
                  aria-label={`Line ${region.id}`}
                  aria-pressed={isSelected}
                  fill={isSelected ? 'rgba(13, 31, 60, 0.18)' : 'rgba(82, 196, 26, 0.15)'}
                  stroke={isSelected ? 'var(--navy, #0d1f3c)' : '#52c41a'}
                  strokeWidth={isSelected ? 2.5 : 2}
                  style={{ pointerEvents: 'all', cursor: 'pointer' }}
                  onClick={() => onSelectRegion(isSelected ? null : region.id)}
                  onKeyDown={(event) => selectWithKeyboard(event, region.id, isSelected)}
                />
              );
            })}
          </svg>
        )}
      </div>
    </PublicZoomSurface>
  );
}
