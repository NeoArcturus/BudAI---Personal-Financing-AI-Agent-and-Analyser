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

export const getUserUuid = (): string | null => {
  const token = getAuthToken();
  if (!token) return null;
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));
    return JSON.parse(jsonPayload).sub || null;
  } catch (e) {
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

let isRefreshing = false;
let failedQueue: { resolve: (value: unknown) => void; reject: (reason?: unknown) => void }[] = [];

const processQueue = (error: Error | null, token: string | null = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    if (axios.isAxiosError(error) && error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(token => {
          originalRequest.headers['Authorization'] = 'Bearer ' + token;
          return apiClient(originalRequest);
        }).catch(err => {
          return Promise.reject(err);
        });
      }
      
      originalRequest._retry = true;
      isRefreshing = true;
      
      try {
        const refreshResponse = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {}, {
          withCredentials: true
        });
        const newToken = refreshResponse.data.token;
        if (typeof window !== "undefined") {
          localStorage.setItem("budai_token", newToken);
          
          document.cookie = `budai_token=${newToken}; path=/; max-age=604800; samesite=lax`;
        }
        apiClient.defaults.headers.common['Authorization'] = 'Bearer ' + newToken;
        originalRequest.headers['Authorization'] = 'Bearer ' + newToken;
        processQueue(null, newToken);
        return apiClient(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError as Error, null);
        clearSession();
        if (typeof window !== "undefined") {
          window.dispatchEvent(new Event("budai-unauthorized"));
        }
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
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
