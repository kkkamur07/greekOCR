type ValidationErrorItem = {
  loc?: Array<string | number>;
  msg?: string;
};

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
  const error = (body as { error?: { details?: unknown; message?: unknown } }).error;
  if (error && Array.isArray(error.details)) {
    const detail = formatValidationDetails(error.details);
    if (detail) {
      return detail;
    }
  }
  if (error && typeof error.message === 'string') {
    return error.message;
  }
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return formatValidationDetails(detail) || 'Request failed';
  }
  return 'Request failed';
}

function formatValidationDetails(details: unknown[]): string {
  return details
    .map((item) => {
      const err = item as ValidationErrorItem;
      const message = err.msg ?? String(item);
      const location = Array.isArray(err.loc) ? err.loc.join('.') : '';
      return location ? `${location}: ${message}` : message;
    })
    .join('; ');
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
