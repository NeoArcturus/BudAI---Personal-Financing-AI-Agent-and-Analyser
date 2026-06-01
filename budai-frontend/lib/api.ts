"use client";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8080";

export const getApiUrl = (path: string): string => {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
};

export const getAuthToken = (): string => {
  if (typeof window === "undefined") return "";
  return localStorage.getItem("budai_token") || "";
};

export const clearSession = (): void => {
  if (typeof window !== "undefined") {
    // Clear only sensitive session data
    localStorage.removeItem("budai_token");
    
    // We preserve budai_user_name and budai_widgets_* to allow 
    // for preference restoration on next login on this device.
    
    document.cookie =
      "budai_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax";
  }
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
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("budai-unauthorized"));
    }
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

export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  withAuth = false,
): Promise<T> {
  const response = await apiFetch(path, init, withAuth);
  return response.json() as Promise<T>;
}
