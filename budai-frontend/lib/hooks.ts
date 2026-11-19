import React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Account, Transaction, BankChartData } from "@/types";
import { useBudAIWebSocket } from "@/app/context/WebSocketProvider";

import { useState, useEffect } from "react";

export function usePersistedState<T>(key: string, defaultValue: T): [T, (val: T) => void] {
  const [state, setState] = useState<T>(defaultValue);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const userName = localStorage.getItem("budai_user_name") || "User";
    const storageKey = `budai_pref_${userName}_${key}`;
    const saved = localStorage.getItem(storageKey);
    let parsed = defaultValue;
    
    if (saved !== null) {
      try {
        parsed = JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse persisted state", e);
      }
    }
    
    const timeoutId = setTimeout(() => {
      setState(parsed);
      setIsLoaded(true);
    }, 0);

    return () => clearTimeout(timeoutId);
  }, [key, defaultValue]);

  useEffect(() => {
    if (!isLoaded || typeof window === "undefined") return;
    const userName = localStorage.getItem("budai_user_name") || "User";
    const storageKey = `budai_pref_${userName}_${key}`;
    localStorage.setItem(storageKey, JSON.stringify(state));
  }, [key, state, isLoaded]);

  return [state, setState];
}

export function useAccounts(initialData?: Account[]) {
  return useQuery({
    queryKey: ["accounts"],
    queryFn: async () => {
      const res = await apiFetch("/api/accounts", {}, true);
      const data = (await res.json()) as { accounts?: Account[] };
      return data.accounts || [];
    },
    initialData: initialData && initialData.length > 0 ? initialData : undefined,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export interface ConnectionMetadataItem {
  bank_uuid: string;
  bank_name: string;
  consent_status: string;
  needs_reauth: boolean;
  reauth_link: string | null;
  metadata?: Record<string, any>;
}

export function useConnectionMetadata() {
  return useQuery({
    queryKey: ["connections-metadata"],
    queryFn: async (): Promise<ConnectionMetadataItem[]> => {
      const res = await apiFetch("/api/auth/connections/metadata", {}, true);
      if (!res.ok) return [];
      const data = (await res.json()) as { results?: ConnectionMetadataItem[] };
      return data.results || [];
    },
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useTransactions(
  accountId: string,
  from?: string,
  to?: string,
  initialData?: Transaction[],
) {
  return useQuery({
    queryKey: ["transactions", accountId, from, to],
    queryFn: async (): Promise<Transaction[]> => {
      const queryParams = new URLSearchParams();
      if (from) queryParams.append("from", from);
      if (to) queryParams.append("to", to);
      const queryStr = queryParams.toString()
        ? `?${queryParams.toString()}`
        : "";

      const res = await apiFetch(
        `/api/accounts/${accountId}/transactions${queryStr}`,
        {},
        true,
      );

      if (res.status === 202) {
        return [];
      }

      const data = (await res.json()) as { transactions?: Transaction[] };
      return data.transactions || [];
    },
    initialData,
    enabled: !!accountId && !accountId.startsWith("react-aria-"),
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useDashboardSnapshot(
  days: number = 30,
  page: number = 1,
  limit: number = 50,
  initialData?: any,
) {
  return useQuery({
    queryKey: ["dashboard-snapshot", days, page, limit],
    queryFn: async () => {
      const res = await apiFetch(
        `/api/dashboard/snapshot?days=${days}&page=${page}&limit=${limit}`,
        {},
        true,
      );
      if (!res.ok) throw new Error("Failed to fetch dashboard snapshot");
      return res.json();
    },
    initialData,
    placeholderData: (previousData) => previousData,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useSearchTransactions(query: string) {
  return useQuery({
    queryKey: ["transactions-search", query],
    queryFn: async () => {
      if (!query || query.length < 2) return [];
      const res = await apiFetch(
        `/api/accounts/transactions/search?q=${encodeURIComponent(query)}`,
        {},
        true,
      );
      const data = await res.json() as { transactions?: Transaction[] };
      return data.transactions || [];
    },
    enabled: query.length >= 2,
    staleTime: 1000 * 60,
  });
}

export function useSpendingTrends(
  accountId: string,
  from: string,
  to: string,
  timeType: "monthly" | "weekly" | "daily" = "monthly",
  initialData?: BankChartData[],
  isReady: boolean = true,
) {
  return useQuery({
    queryKey: ["spending-trends", accountId, from, to, timeType],
    queryFn: async (): Promise<BankChartData[]> => {
      if (!accountId || accountId.startsWith("react-aria-")) return [];
      const queryParams = new URLSearchParams();
      if (from) queryParams.append("from", from);
      if (to) queryParams.append("to", to);
      const res = await apiFetch(
        `/api/accounts/${accountId}/transactions?${queryParams.toString()}`,
        {},
        true,
      );
      if (!res.ok) return [];
      const data = (await res.json()) as { transactions?: Transaction[] };
      const txs = data.transactions || [];
      const map: Record<string, number> = {};
      txs.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0) {
          const dateStr = tx.timestamp ? tx.timestamp.split("T")[0] : "";
          if (dateStr) {
            map[dateStr] = (map[dateStr] || 0) + Math.abs(amt);
          }
        }
      });
      const dataPoints = Object.entries(map).map(([Date, Amount]) => ({
        Date,
        Amount: Math.round(Amount * 100) / 100,
      }));
      return [
        {
          bank_name: "Aggregate",
          data: dataPoints,
        },
      ];
    },
    initialData,
    enabled: !!accountId && !accountId.startsWith("react-aria-") && !!from && !!to && isReady,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useExpenseCategories(
  accountId: string,
  from: string,
  to: string,
  isReady: boolean = true,
) {
  return useQuery({
    queryKey: ["expense-categories", accountId, from, to],
    queryFn: async (): Promise<BankChartData[]> => {
      if (!accountId || accountId.startsWith("react-aria-")) return [];
      const queryParams = new URLSearchParams();
      if (from) queryParams.append("from", from);
      if (to) queryParams.append("to", to);
      const res = await apiFetch(
        `/api/accounts/${accountId}/transactions?${queryParams.toString()}`,
        {},
        true,
      );
      if (!res.ok) return [];
      const data = (await res.json()) as { transactions?: Transaction[] };
      const txs = data.transactions || [];
      const catMap: Record<string, number> = {};
      txs.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0) {
          const cat = tx.category || "General Spending";
          catMap[cat] = (catMap[cat] || 0) + Math.abs(amt);
        }
      });
      const dataPoints = Object.entries(catMap)
        .map(([Category, Total_Amount]) => ({
          Category,
          Total_Amount: Math.round(Total_Amount * 100) / 100,
        }))
        .sort((a, b) => (b.Total_Amount || 0) - (a.Total_Amount || 0));

      return [
        {
          bank_name: "Categories",
          data: dataPoints,
        },
      ];
    },
    enabled: !!accountId && !accountId.startsWith("react-aria-") && !!from && !!to && isReady,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useChatSessions() {
  return useQuery({
    queryKey: ["chat-sessions"],
    queryFn: async () => {
      const res = await apiFetch("/api/chat/sessions", {}, true);
      const data = await res.json() as unknown;
      return Array.isArray(data) ? data : [];
    },
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useAdvisorInsight(widgetId: string, _contextData?: unknown) {
  return useQuery({
    queryKey: ["advisor-insight", widgetId],
    queryFn: async () => {
      return "";
    },
    enabled: false,
  });
}

export function useUserProfile() {
  return useQuery({
    queryKey: ["user-profile"],
    queryFn: async () => {
      const res = await apiFetch("/api/auth/me", {}, true);
      const data = (await res.json()) as { is_onboarded?: boolean; user_uuid?: string };
      return data;
    },
    refetchInterval: (query) => {
      return query.state.data?.is_onboarded ? false : 5000;
    },
    staleTime: 1000 * 60 * 10,
    refetchOnWindowFocus: false,
  });
}

export function useBuckets() {
  return useQuery({
    queryKey: ["buckets"],
    queryFn: async () => {
      const res = await apiFetch("/api/buckets/", {}, true);
      if (!res.ok) return [];
      const data = (await res.json()) as { status?: string; buckets?: any[] };
      return data.buckets || [];
    },
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

export function useSubscriptions(userUuid?: string) {
  return useQuery({
    queryKey: ["subscriptions", userUuid],
    queryFn: async () => {
      return [];
    },
    enabled: !!userUuid,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
    refetchOnWindowFocus: false,
  });
}

import { parseDate, DateValue, today, getLocalTimeZone } from "@internationalized/date";

export function usePersistedDate(key: string, defaultDate: DateValue | null): [DateValue | null, (val: DateValue | null) => void] {
  const [strValue, setStrValue] = usePersistedState<string | null>(key, defaultDate ? defaultDate.toString() : null);
  
  const dateValue = React.useMemo(() => {
    if (!strValue) return null;
    try {
      return parseDate(strValue.split("T")[0]);
    } catch (e) {
      return defaultDate;
    }
  }, [strValue, defaultDate]);

  const setDateValue = React.useCallback((val: DateValue | null) => {
    setStrValue(val ? val.toString() : null);
  }, [setStrValue]);

  return [dateValue, setDateValue];
}

export function useUpdateTransactionCategory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      transactionUuid,
      category,
    }: {
      transactionUuid: string;
      category: string;
    }) => {
      const res = await apiFetch("/api/categorizer/labels", {
        method: "POST",
        body: JSON.stringify({
          transaction_uuid: transactionUuid,
          corrected_label: category,
          retrain_model: true,
        }),
      });
      if (!res.ok) throw new Error("Failed to update transaction category");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["transactions"] });
      queryClient.invalidateQueries({ queryKey: ["expense-categories"] });
      queryClient.invalidateQueries({ queryKey: ["spending-trends"] });
    },
  });
}

export function useNotificationActions() {
  const queryClient = useQueryClient();

  const markAsRead = useMutation({
    mutationFn: async (alertId: string) => {
      const res = await apiFetch(`/api/notifications/${alertId}/read`, {
        method: "PATCH",
      });
      if (!res.ok) throw new Error("Failed to mark notice as read");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  const dismissAlert = useMutation({
    mutationFn: async (alertId: string) => {
      const res = await apiFetch(`/api/notifications/${alertId}/dismiss`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to dismiss notice");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  const markAllAsRead = useMutation({
    mutationFn: async () => {
      const res = await apiFetch("/api/notifications/read-all", {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to mark all notices as read");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  return { markAsRead, dismissAlert, markAllAsRead };
}
