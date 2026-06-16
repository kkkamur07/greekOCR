import type { components } from './schema';

type ValidationError = components['schemas']['ValidationError'];

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function formatDetail(body: unknown): string {
  if (!body || typeof body !== 'object') {
    return 'Request failed';
  }
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const err = item as ValidationError;
        return err.msg ?? String(item);
      })
      .join('; ');
  }
  return 'Request failed';
}

export async function parseApiError(response: Response): Promise<ApiError> {
  let message = response.statusText || 'Request failed';
  try {
    const body = await response.json();
    message = formatDetail(body);
  } catch {
    // non-JSON body
  }
  return new ApiError(message, response.status);
}
