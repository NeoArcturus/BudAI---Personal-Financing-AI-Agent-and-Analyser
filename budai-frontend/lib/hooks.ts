import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Account, Transaction, BankChartData } from "@/types";

/**
 * Hook to fetch all connected bank accounts with caching.
 */
export function useAccounts() {
  return useQuery({
    queryKey: ["accounts"],
    queryFn: async () => {
      const res = await apiFetch("/api/accounts", {}, true);
      const data = await res.json();
      return (data.accounts as Account[]) || [];
    },
    staleTime: 1000 * 60 * 5, // Keep in memory for 5 mins
    refetchInterval: 1000 * 60 * 5, // Silent refresh every 5 mins
    refetchOnWindowFocus: false, // Don't hit backend on tab switch
  });
}

/**
 * Hook to fetch transactions for a specific account with caching.
 */
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
      const data = await res.json();
      return (data.transactions as Transaction[]) || [];
    },
    initialData,
    enabled: !!accountId,
    staleTime: 1000 * 60 * 5, // Keep in memory for 5 mins
    refetchInterval: 1000 * 60 * 5, // Silent refresh every 5 mins
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch spending trends for a specific account with caching.
 */
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
      const result = await res.json();
      return (result.data as BankChartData[]) || [];
    },
    initialData,
    enabled: !!accountId && !!from && !!to,
    staleTime: 1000 * 60 * 5, // Keep in memory for 5 mins
    refetchInterval: 1000 * 60 * 5, // Silent refresh every 5 mins
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to get AI Advisor summary for a specific widget context.
 */
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
      const data = await res.json();
      return data.job_id as string;
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
      const data = await res.json();
      if (data.status === "pending") {
        throw new Error("STILL_PENDING");
      }
      if (data.status === "failed") {
        return "Unable to generate insight at this moment.";
      }
      return data.insight as string;
    },
    enabled: !!initJob.data,
    refetchInterval: (query) => {
        return query.state.data ? false : 2000;
    },
    retry: (failureCount, error: any) => {
        if (error.message === "STILL_PENDING") return true;
        return failureCount < 2;
    },
  });
}
