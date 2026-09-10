/**
 * Typed client for the FastAPI backend.
 *
 * The token lives in `localStorage` and is read at request time rather than
 * captured once, so a login or logout in one tab takes effect on the next
 * request everywhere without rebuilding the client.
 *
 * A note on that choice: `localStorage` is readable by any script on the page,
 * so this is not the storage a bank would use — an httpOnly cookie set by the
 * server would be. It is the pragmatic option for a token-based demo API with
 * no CSRF machinery, and it is called out here rather than left to look like an
 * oversight.
 */

import type {
  FoodItemDetail,
  FoodItemSummary,
  ImpressionBatchResponse,
  MenuQuery,
  OrderResponse,
  Page,
  RatingResponse,
  RatingWithItem,
  RecommendationResponse,
  RegisterPayload,
  TasteProfile,
  TokenResponse,
  UserResponse,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

const TOKEN_KEY = "foodrec.token";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }

  get isUnauthorised(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing and hardened settings can make storage throw outright.
    return null;
  }
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (token === null) window.localStorage.removeItem(TOKEN_KEY);
    else window.localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* Nothing useful to do; the session simply will not persist a reload. */
  }
}

/** Turn FastAPI's error shapes into a single readable message. */
function readErrorDetail(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    // 422 from Pydantic is a list of per-field errors.
    if (Array.isArray(detail)) {
      const messages = detail
        .map((entry) => {
          if (typeof entry === "object" && entry !== null && "msg" in entry) {
            const field = Array.isArray((entry as { loc?: unknown[] }).loc)
              ? ((entry as { loc: unknown[] }).loc.slice(-1)[0] as string)
              : null;
            const msg = String((entry as { msg: unknown }).msg);
            return field ? `${field}: ${msg}` : msg;
          }
          return null;
        })
        .filter(Boolean);
      if (messages.length) return messages.join(", ");
    }
  }
  return `Request failed (${status})`;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Send the bearer token if one is stored. */
  auth?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, signal, auth = false } = options;

  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    // A network-level failure is far more likely to be "the API is not running"
    // than anything else during development, so say so.
    throw new ApiError(
      0,
      `Could not reach the API at ${API_BASE}. Is the backend running?`,
    );
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const parsed: unknown = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new ApiError(response.status, readErrorDetail(parsed, response.status));
  }
  return parsed as T;
}

function toQuery(params: Record<string, unknown>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      // The menu endpoint takes `cuisine` repeated, not comma-joined.
      for (const entry of value) search.append(key, String(entry));
    } else {
      search.append(key, String(value));
    }
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

export const api = {
  register(payload: RegisterPayload) {
    return request<TokenResponse>("/api/auth/register", {
      method: "POST",
      body: payload,
    });
  },

  login(email: string, password: string) {
    return request<TokenResponse>("/api/auth/login", {
      method: "POST",
      body: { email, password },
    });
  },

  me() {
    return request<UserResponse>("/api/auth/me", { auth: true });
  },

  tasteProfile() {
    return request<TasteProfile>("/api/auth/me/taste-profile", { auth: true });
  },

  menu(query: MenuQuery = {}) {
    return request<Page<FoodItemSummary>>(`/api/menu${toQuery({ ...query })}`);
  },

  menuItem(id: number) {
    return request<FoodItemDetail>(`/api/menu/${id}`);
  },

  cuisines() {
    return request<string[]>("/api/menu/cuisines");
  },

  recommendationsForMe(limit = 12) {
    return request<RecommendationResponse>(
      `/api/recommendations/for-me${toQuery({ limit })}`,
      { auth: true },
    );
  },

  popular(limit = 12) {
    return request<RecommendationResponse>(
      `/api/recommendations/popular${toQuery({ limit })}`,
    );
  },

  similar(itemId: number, limit = 8) {
    return request<RecommendationResponse>(
      `/api/recommendations/similar/${itemId}${toQuery({ limit })}`,
    );
  },

  rate(foodItemId: number, rating: number) {
    return request<RatingResponse>("/api/ratings", {
      method: "POST",
      auth: true,
      body: { food_item_id: foodItemId, rating },
    });
  },

  myRatings(limit = 50, offset = 0) {
    return request<Page<RatingWithItem>>(
      `/api/ratings/me${toQuery({ limit, offset })}`,
      { auth: true },
    );
  },

  createOrder(items: Array<{ food_item_id: number; quantity: number }>) {
    return request<OrderResponse>("/api/orders", {
      method: "POST",
      auth: true,
      body: { items },
    });
  },

  orderHistory(limit = 20, offset = 0) {
    return request<Page<OrderResponse>>(
      `/api/orders/history${toQuery({ limit, offset })}`,
      { auth: true },
    );
  },

  logImpressions(
    impressions: Array<{ food_item_id: number; was_ordered?: boolean }>,
  ) {
    return request<ImpressionBatchResponse>("/api/impressions/batch", {
      method: "POST",
      auth: true,
      body: { impressions },
    });
  },
};
