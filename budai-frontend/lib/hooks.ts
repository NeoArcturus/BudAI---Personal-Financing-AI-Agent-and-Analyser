import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Account, Transaction } from "@/types";

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
  });
}

/**
 * Hook to fetch transactions for a specific account with caching.
 */
export function useTransactions(accountId: string, from?: string, to?: string) {
  return useQuery({
    queryKey: ["transactions", accountId, from, to],
    queryFn: async () => {
      const queryParams = new URLSearchParams();
      if (from) queryParams.append("from", from);
      if (to) queryParams.append("to", to);
      const queryStr = queryParams.toString() ? `?${queryParams.toString()}` : "";

      const res = await apiFetch(`/api/accounts/${accountId}/transactions${queryStr}`, {}, true);
      const data = await res.json();
      return (data.transactions as Transaction[]) || [];
    },
    enabled: !!accountId,
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
      return (result.data as any[]) || [];
    },
    enabled: !!accountId && !!from && !!to,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

/**
 * Hook to get AI Advisor summary for a specific widget context.
 * Note: Uses placeholder logic until /api/advisor/summarize exists.
 */
export function useAdvisorInsight(widgetId: string, contextData: unknown) {
  return useQuery({
    queryKey: ["advisor-insight", widgetId, JSON.stringify(contextData)],
    queryFn: async () => {

      await new Promise((resolve) => setTimeout(resolve, 1500));

      if (widgetId.includes("cashFlow")) {
        return "Your income has remained stable, but I've detected a 12% rise in subscription services this month. Canceling unused plans could save you £45 monthly.";
      }
      if (widgetId.includes("spendingTrend")) {
        return "Spending is trending downward in discretionary categories, which is excellent. You are currently on track to reach your savings goal 3 weeks ahead of schedule.";
      }
      return "I've analyzed your latest transaction patterns. Overall liquidity is healthy, though your emergency fund allocation could be optimized for higher yields.";
    },
    enabled: !!contextData,
    staleTime: 1000 * 60 * 30,
  });
}
