/**
 * Client-side page-image normalization: decode any supported raster to WebP.
 *
 * The production API is serverless, and Vercel Functions cap a request body at
 * 4.5 MB - far below a manuscript page scan. Instead of POSTing the bytes through
 * the API (which would be rejected), the browser re-encodes the image to WebP here
 * and then PUTs the WebP straight to object storage using a presigned URL the API
 * minted. The API still validates and re-reads dimensions on finalize, so this is
 * an optimization that moves bytes around, not a place where validation is skipped.
 *
 * The server's own hard bound is 200 MP (see ``MAX_DECODE_PIXELS``); we keep the
 * client bound tighter so a page the user can still upload is one the server will
 * accept, and so an enormous scan fails fast before a canvas allocation.
 */

export const MAX_PART_DECODE_PIXELS = 200_000_000;

export type EncodedPartImage = {
  /** WebP bytes. */
  data: Blob;
  /** Source width in pixels. */
  width: number;
  /** Source height in pixels. */
  height: number;
};

export class UnsupportedImageError extends Error {
  constructor(message = "Uploaded file is not a valid image") {
    super(message);
    this.name = "UnsupportedImageError";
  }
}

/**
 * Decode a File to a bitmap and re-encode it as a WebP Blob with its dimensions.
 *
 * ``createImageBitmap`` decodes in the background; browsers that lack it (older
 * Safari) reject here, and the caller falls back to the multipart upload. The
 * re-encode draws the bitmap to a canvas at its natural resolution - no resize -
 * then asks for ``image/webp`` output, which is what the media store expects.
 */
export async function encodePartImage(file: File): Promise<EncodedPartImage> {
  const bitmap = await createImageBitmap(file).catch(() => {
    throw new UnsupportedImageError();
  });

  const width = bitmap.width;
  const height = bitmap.height;
  if (width * height > MAX_PART_DECODE_PIXELS) {
    bitmap.close();
    throw new UnsupportedImageError("Uploaded image is too large");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    throw new UnsupportedImageError();
  }

  // A transparent or palette-with-alpha source must stay RGBA; everything else
  // is flattened to opaque RGB so a scanned page has no alpha channel in storage.
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (result) => {
        if (result) resolve(result);
        else
          reject(new UnsupportedImageError("Could not encode image to WebP"));
      },
      "image/webp",
      0.95,
    );
  });

  return { data: blob, width, height };
}
