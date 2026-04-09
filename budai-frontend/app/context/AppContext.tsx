"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Account,
  NativeChartConfig,
  TabType,
  ToolParameters,
  BankChartData,
} from "@/types";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { apiFetch } from "@/lib/api";

interface BudAIContextType {
  accounts: Account[];
  activeAccountId: string | null;
  setActiveAccountId: (id: string | null) => void;
  chartConfig: NativeChartConfig | null;
  setChartConfig: (config: NativeChartConfig | null) => void;
  isGenerating: boolean;
  setIsGenerating: (val: boolean) => void;
  handleAiChartTrigger: (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
    extraParam?: string,
  ) => Promise<void>;
  userName: string;
  totalBalance: number;
}

const BudAIContext = createContext<BudAIContextType | undefined>(undefined);

export const BudAIProvider = ({ children }: { children: React.ReactNode }) => {
  const router = useRouter();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [activeAccountId, setActiveAccountId] = useState<string | null>("ALL");
  const [chartConfig, setChartConfig] = useState<NativeChartConfig | null>(
    null,
  );
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [userName] = useState<string>(
    typeof window !== "undefined"
      ? localStorage.getItem("budai_user_name") || "User"
      : "User",
  );

  useEffect(() => {
    const token = localStorage.getItem("budai_token");
    if (!token) {
      router.push("/login");
      return;
    }
    apiFetch("/api/accounts/", {}, true)
      .then((res) => res.json())
      .then((data) => {
        if (data.accounts) setAccounts(data.accounts);
      })
      .catch(() => {});
  }, [router]);

  const handleAiChartTrigger = async (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
    extraParam?: string,
  ) => {
    setIsGenerating(true);
    let targetId = aiTargetId || activeAccountId;
    let isCacheToken = false;

    if (aiTargetId && aiTargetId.startsWith("CACHE_")) {
      targetId = aiTargetId;
      isCacheToken = true;
    } else if (aiTargetId && aiTargetId !== "ALL") {
      const targetBanks = aiTargetId.split(",").map((b) => b.trim());
      const resolvedIds: string[] = [];
      targetBanks.forEach((bank) => {
        const matched = accounts.find(
          (a) =>
            a.account_id === bank ||
            a.bank_name?.toLowerCase() === bank.toLowerCase() ||
            a.provider_name?.toLowerCase() === bank.toLowerCase(),
        );
        if (matched) resolvedIds.push(matched.account_id);
        else resolvedIds.push(bank);
      });
      targetId = resolvedIds.join(",");
    }

    if (!targetId) {
      setIsGenerating(false);
      return;
    }

    if (
      !isCacheToken &&
      targetId !== activeAccountId &&
      !targetId.includes(",")
    ) {
      setActiveAccountId(targetId);
    }

    let toolName = "";
    const params: ToolParameters = { bank_name_or_id: targetId };

    if (type === "categorized" || type === "categorized_doughnut") {
      toolName = "classify_financial_data";
      if (extraParam && extraParam.includes("|")) {
        const [fromDate, toDate] = extraParam.split("|");
        params.from_date = fromDate;
        params.to_date = toDate;
      } else {
        params.from_date = "2024-01-01";
        params.to_date = new Date().toISOString().split("T")[0];
      }
    } else if (type === "expense_forecast") {
      toolName = "generate_expense_forecast";
      params.days = extraParam ? parseInt(extraParam, 10) : 30;
    } else if (type === "balance_forecast") {
      toolName = "generate_financial_forecast";
      params.days = extraParam ? parseInt(extraParam, 10) : 60;
    } else if (type === "cash_flow_mixed") {
      toolName = "plot_cash_flow_mixed";
    } else if (type === "health_radar") {
      toolName = "plot_health_radar";
    } else if (type.startsWith("historical")) {
      toolName = "plot_expenses";
      params.plot_time_type = type.split("_")[1] || "monthly";
      if (extraParam && extraParam.includes("|")) {
        const [fromDate, toDate] = extraParam.split("|");
        if (fromDate) params.from_date = fromDate;
        if (toDate) params.to_date = toDate;
      }
    }

    if (!toolName) {
      setIsGenerating(false);
      return;
    }

    try {
      // const res = await apiFetch(
      //   "/api/media/execute",
      //   {
      //   method: "POST",
      //   body: JSON.stringify({ tool_name: toolName, parameters: params }),
      //   },
      //   true,
      // );
      const res = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          body: JSON.stringify({ tool_name: toolName, parameters: params }),
        },
        true,
      );
      if (!res.ok) throw new Error("Chart generation failed");
      const jsonRes = (await res.json()) as { data: BankChartData[] };
      const newConfig = buildChartConfig(
        type,
        jsonRes.data || [],
        params,
        customTitle,
      );
      if (newConfig) {
        setChartConfig(newConfig);
        router.push("/finances");
      }
    } catch (err) {
      console.log(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const totalBalance = accounts.reduce(
    (sum, acc) => sum + (acc.balance ?? acc.account_balance ?? 0),
    0,
  );

  return (
    <BudAIContext.Provider
      value={{
        accounts,
        activeAccountId,
        setActiveAccountId,
        chartConfig,
        setChartConfig,
        isGenerating,
        setIsGenerating,
        handleAiChartTrigger,
        userName,
        totalBalance,
      }}
    >
      {children}
    </BudAIContext.Provider>
  );
};

export const useBudAI = () => {
  const context = useContext(BudAIContext);
  if (!context) throw new Error("useBudAI must be used within BudAIProvider");
  return context;
};
