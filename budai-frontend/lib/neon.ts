import { apiFetch } from "./api";

export interface NeonServiceStatus {
  branch: string;
  services: {
    postgres: {
      configured: boolean;
      connected: boolean;
      version_summary?: string | null;
    };
    auth: {
      configured: boolean;
      base_url?: string | null;
      jwks_url?: string | null;
    };
    object_storage: {
      configured: boolean;
      endpoint?: string | null;
      bucket?: string | null;
      objects_cached: number;
    };
    functions: {
      configured: boolean;
      hello_url?: string | null;
      invocation_test?: Record<string, unknown> | null;
    };
  };
}

export const getNeonStatus = async (): Promise<NeonServiceStatus> => {
  const res = await apiFetch("/api/neon/status");
  if (!res.ok) {
    throw new Error(`Failed to fetch Neon status: ${res.statusText}`);
  }
  return (await res.json()) as NeonServiceStatus;
};

export const invokeNeonHello = async () => {
  const res = await apiFetch("/api/neon/functions/hello");
  if (!res.ok) {
    throw new Error(`Failed to invoke Neon hello function: ${res.statusText}`);
  }
  return res.json();
};

export const listNeonStorageObjects = async (prefix = "") => {
  const res = await apiFetch(`/api/neon/storage/list?prefix=${encodeURIComponent(prefix)}`);
  if (!res.ok) {
    throw new Error(`Failed to list Neon storage objects: ${res.statusText}`);
  }
  return res.json();
};

export const uploadToNeonStorage = async (file: File, pathPrefix = "uploads") => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("path_prefix", pathPrefix);

  const res = await apiFetch("/api/neon/storage/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Failed to upload to Neon storage: ${res.statusText}`);
  }
  return res.json();
};
