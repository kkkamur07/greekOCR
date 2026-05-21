import { useEffect, useState } from 'react';
import { Image, Spin } from 'antd';
import { API_BASE_URL } from '../api/client';
import { getAccessToken } from '../auth/storage';

export function AuthenticatedImage({
  src,
  alt,
  width = 120,
}: {
  src: string;
  alt: string;
  width?: number;
}) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let objectUrl: string | null = null;
    const fullUrl = src.startsWith('http') ? src : `${API_BASE_URL}${src}`;
    const token = getAccessToken();

    (async () => {
      try {
        const res = await fetch(fullUrl, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) {
          setBlobUrl(null);
          return;
        }
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      } finally {
        setLoading(false);
      }
    })();

    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [src]);

  if (loading) {
    return <Spin size="small" />;
  }
  if (!blobUrl) {
    return <span>—</span>;
  }
  return <Image src={blobUrl} alt={alt} width={width} preview />;
}
