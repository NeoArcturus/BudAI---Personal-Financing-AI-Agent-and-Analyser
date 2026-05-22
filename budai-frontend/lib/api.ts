"use client";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8080";

export const getApiUrl = (path: string): string => {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
};

export const getAuthToken = (): string => {
  return localStorage.getItem("budai_token") || "";
};

export const clearSession = (): void => {
  localStorage.removeItem("budai_token");
  localStorage.removeItem("budai_user_name");
  document.cookie = "budai_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
};

export async function apiFetch(
  path: string,
  init?: RequestInit,
  withAuth = false,
): Promise<Response> {
  const headers = new Headers(init?.headers || {});
  if (!headers.has("Content-Type") && init?.body) {
    headers.set("Content-Type", "application/json");
  }
  if (withAuth) {
    const token = getAuthToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(getApiUrl(path), { ...init, headers });
  if (response.status === 401 && withAuth) {
    clearSession();
    if (typeof window !== "undefined") window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.message ||
        `API Error: ${response.status} ${response.statusText}`,
    );
  }

  return response;
}

/**
 * Typed wrapper for apiFetch to handle JSON parsing automatically
 */
export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  withAuth = false,
): Promise<T> {
  const response = await apiFetch(path, init, withAuth);
  return response.json() as Promise<T>;
}
