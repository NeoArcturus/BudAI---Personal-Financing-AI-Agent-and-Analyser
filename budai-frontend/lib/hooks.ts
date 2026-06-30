import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Account, Transaction, BankChartData } from "@/types";

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

export function useAccounts() {
  return useQuery({
    queryKey: ["accounts"],
    queryFn: async () => {
      const res = await apiFetch("/api/accounts", {}, true);
      const data = await res.json() as { accounts?: Account[] };
      return data.accounts || [];
    },
    staleTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 60 * 5,
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
    queryFn: async () => {
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
      const data = await res.json() as { transactions?: Transaction[] };
      return data.transactions || [];
    },
    initialData,
    enabled: !!accountId,
    staleTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 60 * 5,
    refetchOnWindowFocus: false,
  });
}

export function useSpendingTrends(
  accountId: string,
  from: string,
  to: string,
  timeType: "monthly" | "weekly" | "daily" = "monthly",
  initialData?: BankChartData[],
) {
  return useQuery({
    queryKey: ["spending-trends", accountId, from, to, timeType],
    queryFn: async () => {
      const res = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            tool_name: "plot_expenses",
            parameters: {
              plot_time_type: timeType,
              from_date: from,
              to_date: to,
              bank_name_or_id: accountId,
            },
          }),
        },
        true,
      );
      const result = await res.json() as { data?: BankChartData[] };
      return result.data || [];
    },
    initialData,
    enabled: !!accountId && !!from && !!to,
    staleTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 60 * 5,
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
    refetchInterval: 1000 * 60 * 5,
    refetchOnWindowFocus: false,
  });
}

export function useAdvisorInsight(widgetId: string, contextData: unknown) {
  const initJob = useQuery({
    queryKey: ["init-advisor", widgetId, JSON.stringify(contextData || [])],
    queryFn: async () => {
      const res = await apiFetch("/api/advisor/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          widget_id: widgetId,
          context_data: {
            data: Array.isArray(contextData) ? contextData.slice(0, 20) : contextData,
            timestamp: new Date().toISOString()
          },
        }),
      }, true);
      const data = await res.json() as { job_id: string };
      return data.job_id;
    },
    staleTime: Infinity,
    enabled: !!widgetId,
  });

  return useQuery({
    queryKey: ["advisor-insight-poll", initJob.data],
    queryFn: async () => {
      const res = await apiFetch(`/api/advisor/status/${initJob.data}`, {
        method: "GET",
      }, true);
      const data = await res.json() as { status: string; insight?: string };
      if (data.status === "pending") {
        throw new Error("STILL_PENDING");
      }
      if (data.status === "failed") {
        return "Unable to generate insight at this moment.";
      }
      return data.insight || "";
    },
    enabled: !!initJob.data,
    refetchInterval: (query) => {
        return query.state.data ? false : 2000;
    },
    retry: (failureCount, error: Error) => {
        if (error.message === "STILL_PENDING") return true;
        return failureCount < 2;
    },
  });
}
