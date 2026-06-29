"use client";

import axios, { AxiosRequestConfig } from "axios";

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

export const clearAdviceCache = (): void => {
  if (typeof window !== "undefined") {
    Object.keys(localStorage).forEach((key) => {
      if (key.startsWith("budai_advice_")) {
        localStorage.removeItem(key);
      }
    });
  }
};

export const clearSession = (): void => {
  if (typeof window !== "undefined") {
    localStorage.removeItem("budai_token");
    clearAdviceCache();
    document.cookie =
      "budai_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax";
  }
};

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      clearSession();
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

// Drop-in replacement for the native fetch using axios
export async function apiFetch(
  path: string,
  init?: RequestInit,
  withAuth = false,
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
      } catch (e) {
        console.log("Error:", e);
        break;
      }
    }
    axiosOptions.data = finalData;
  }

  axiosOptions.headers = {
    ...((init?.headers as Record<string, string>) || {}),
  };

  if (withAuth) {
    const token = getAuthToken();
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
