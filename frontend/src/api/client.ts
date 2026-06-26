import type {
  AnswerResponse,
  EditableSettings,
  GraphData,
  IngestionReport,
  Settings,
  Stats,
} from './types';

const BASE_URL = '/api';

class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch {
    throw new ApiError('Cannot reach the server. Is the backend running?', 0);
  }

  const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (!response.ok) {
    const message =
      typeof data.error === 'string' ? data.error : `Request failed (${response.status})`;
    throw new ApiError(message, response.status);
  }
  return data as T;
}

export const api = {
  ingest: (text: string, title: string): Promise<IngestionReport> =>
    request<IngestionReport>('/documents', {
      method: 'POST',
      body: JSON.stringify({ text, title }),
    }),

  ask: (question: string): Promise<AnswerResponse> =>
    request<AnswerResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  stats: (): Promise<Stats> => request<Stats>('/stats'),

  graph: (limit = 200): Promise<GraphData> => request<GraphData>(`/graph?limit=${limit}`),

  getSettings: (): Promise<Settings> => request<Settings>('/settings'),

  updateSettings: (patch: EditableSettings): Promise<Settings> =>
    request<Settings>('/settings', { method: 'PUT', body: JSON.stringify(patch) }),

  reset: (): Promise<{ status: string }> => request('/reset', { method: 'POST' }),
};

export { ApiError };
