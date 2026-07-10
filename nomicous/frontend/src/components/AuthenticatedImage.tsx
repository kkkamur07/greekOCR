import { useEffect, useState, type CSSProperties, type SyntheticEvent } from 'react';
import { API_BASE_URL } from '../api/client';
import { acquirePartImage, normalizePartImagePath } from '../api/imageCache';

export function resolveProtectedMediaUrl(src: string): string | null {
  const path = normalizePartImagePath(src);
  return path ? new URL(path, `${API_BASE_URL}/`).toString() : null;
}

export function AuthenticatedImage({
  src,
  alt,
  width = 120,
  compact = false,
  onLoad,
  style,
  className,
}: {
  src: string;
  alt: string;
  width?: number;
  compact?: boolean;
  onLoad?: (event: SyntheticEvent<HTMLImageElement>) => void;
  style?: CSSProperties;
  className?: string;
}) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let release: (() => void) | null = null;

    setLoading(true);
    setFailed(false);
    setBlobUrl(null);

    void (async () => {
      try {
        const image = await acquirePartImage(src);
        release = image.release;
        if (cancelled) {
          image.release();
          return;
        }
        setBlobUrl(image.objectUrl);
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      release?.();
    };
  }, [src]);

  if (loading) {
    return (
      <span
        className={`auth-image auth-image--loading${className ? ` ${className}` : ''}`}
        style={style}
        aria-busy="true"
        aria-label={`Loading ${alt}`}
      />
    );
  }

  if (failed || !blobUrl) {
    return (
      <span
        className={`auth-image auth-image--failed${className ? ` ${className}` : ''}`}
        style={style}
        aria-label={`${alt} unavailable`}
        title="Image could not be loaded"
      />
    );
  }

  if (compact) {
    return (
      <img
        src={blobUrl}
        alt={alt}
        className={className}
        onLoad={onLoad}
        style={style}
        decoding="async"
      />
    );
  }

  return (
    <img
      src={blobUrl}
      alt={alt}
      className={className}
      width={width}
      onLoad={onLoad}
      style={style}
      decoding="async"
    />
  );
}
