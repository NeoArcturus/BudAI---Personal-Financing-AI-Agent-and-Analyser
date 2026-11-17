"use client";

import axios, { AxiosRequestConfig } from "axios";
import { authClient } from "@/lib/auth/client";

export const getApiBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const override = localStorage.getItem("budai_api_target");
    if (override === "local") return "http://localhost:8080";
  }
  return (
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
    "http://localhost:8080"
  );
};

export const setApiTarget = (target: "local" | "prod" | "auto"): void => {
  if (typeof window === "undefined") return;
  if (target === "auto") {
    localStorage.removeItem("budai_api_target");
  } else {
    localStorage.setItem("budai_api_target", target);
  }
  window.location.reload();
};

export const API_BASE_URL = getApiBaseUrl();

export const getApiUrl = (path: string): string => {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalized}`;
};

export const notifyAuthChange = (): void => {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event("budai-auth-change"));
  }
};

let cachedJwt: string | null = null;
let tokenExpiry = 0;

export const getAuthTokenAsync = async (): Promise<string> => {
  if (typeof window === "undefined") return "";

  // Return cached JWT if still valid (refresh 1 minute before 15m expiration)
  if (cachedJwt && Date.now() < tokenExpiry) {
    return cachedJwt;
  }

  try {
    const res = await (authClient as any).token?.();
    const token = res?.data?.token || res?.token;
    if (token) {
      cachedJwt = token;
      tokenExpiry = Date.now() + 14 * 60 * 1000;
      return token;
    }
  } catch {
    // Fall back to stored session tokens
  }

  return getAuthToken();
};

export const getAuthToken = (): string => {
  if (typeof window === "undefined") return "";
  if (cachedJwt) return cachedJwt;

  const local = localStorage.getItem("budai_token");
  if (local) return local;

  // Secondary cookie fallback (__Secure-neon-auth.session_token, better-auth.session_token, or budai_token)
  const match = document.cookie.match(/(?:^|;\s*)(?:__Secure-neon-auth\.session_token|better-auth\.session_token|budai_token)=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
};

export const getUserUuid = (): string | null => {
  const token = getAuthToken();
  if (!token) return null;
  try {
    const base64Url = token.split('.')[1];
    if (!base64Url) return null;
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));
    return JSON.parse(jsonPayload).sub || null;
  } catch {
    return null;
  }
};

export const clearAdviceCache = (): void => {
  if (typeof window !== "undefined") {
    Object.keys(localStorage).forEach((key) => {
      if (key.startsWith("budai_advice_")) {
        localStorage.removeItem(key);
      }
    });
  }
};

export const clearSession = async (): Promise<void> => {
  if (typeof window !== "undefined") {
    cachedJwt = null;
    tokenExpiry = 0;
    localStorage.removeItem("budai_token");
    localStorage.removeItem("budai_refresh_token");
    localStorage.removeItem("budai_user_name");
    clearAdviceCache();
    document.cookie = "budai_token=; path=/; max-age=0";
    document.cookie = "better-auth.session_token=; path=/; max-age=0";
    document.cookie = "__Secure-neon-auth.session_token=; path=/; max-age=0";
    try {
      await authClient.signOut();
    } catch {}
    notifyAuthChange();
  }
};

export const apiClient = axios.create({
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use(async (config) => {
  config.baseURL = getApiBaseUrl();
  const token = await getAuthTokenAsync();
  if (token && !config.headers.Authorization) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (
      axios.isAxiosError(error) &&
      (error.response?.status === 401 || error.response?.status === 403) &&
      !originalRequest._retry
    ) {
      originalRequest._retry = true;
      cachedJwt = null;
      tokenExpiry = 0;

      try {
        const freshToken = await getAuthTokenAsync();
        if (freshToken) {
          originalRequest.headers.Authorization = `Bearer ${freshToken}`;
          return apiClient(originalRequest);
        }
      } catch {
        // Fall through to logout
      }

      await clearSession();
      if (typeof window !== "undefined") {
        window.dispatchEvent(new Event("budai-unauthorized"));
      }
    }

    return Promise.reject(error);
  },
);

interface MockResponse {
  ok: boolean;
  status: number;
  statusText: string;
  json: () => Promise<unknown>;
}

export async function apiFetch(
  path: string,
  init?: RequestInit,
  withAuth = true,
): Promise<MockResponse> {
  const axiosOptions: AxiosRequestConfig = {
    method: init?.method || "GET",
  };

  if (init?.body) {
    let finalData: unknown = init.body;
    while (typeof finalData === "string") {
      try {
        const parsed: unknown = JSON.parse(finalData);
        if (typeof parsed === "object" && parsed !== null) {
          finalData = parsed;
        } else {
          break;
        }
      } catch {
        break;
      }
    }
    axiosOptions.data = finalData;
  }

  axiosOptions.headers = {
    ...((init?.headers as Record<string, string>) || {}),
  };

  if (withAuth) {
    const token = await getAuthTokenAsync();
    if (token) {
      axiosOptions.headers.Authorization = `Bearer ${token}`;
    }
  }

  try {
    const url = path.startsWith("http") ? path : getApiUrl(path);
    const response = await apiClient({ url, ...axiosOptions });

    return {
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      statusText: response.statusText,
      json: async () => response.data,
    };
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      if (error.response.status === 401 || error.response.status === 403) {
        await clearSession();
        if (typeof window !== "undefined") {
          window.dispatchEvent(new Event("budai-unauthorized"));
          window.location.href = "/login";
        }
      }

      const errorData = error.response.data || {};
      throw new Error(
        (errorData as { message?: string }).message ||
          `API Error: ${error.response.status} ${error.response.statusText}`,
      );
    }
    if (error instanceof Error) {
      throw new Error(error.message || "Network Error");
    }
    throw new Error("Network Error");
  }
}

export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  withAuth = false,
): Promise<T> {
  const response = await apiFetch(path, init, withAuth);
  return (await response.json()) as T;
}
