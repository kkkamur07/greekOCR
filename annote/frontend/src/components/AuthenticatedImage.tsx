import { useEffect, useState, type CSSProperties, type SyntheticEvent } from 'react';
import { API_BASE_URL } from '../api/client';
import { getAccessToken } from '../auth/storage';

function resolveMediaUrl(src: string): string {
  if (src.startsWith('http')) return src;
  return `${API_BASE_URL}${src}`;
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
    let objectUrl: string | null = null;

    setLoading(true);
    setFailed(false);
    setBlobUrl(null);

    const fullUrl = resolveMediaUrl(src);
    const token = getAccessToken();

    void (async () => {
      try {
        const res = await fetch(fullUrl, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (cancelled) return;
        if (!res.ok) {
          setFailed(true);
          return;
        }
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
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
