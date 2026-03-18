// app/dashboard/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import BudAIChat from "./_components/BudAIChat";
import TransactionModal from "./_components/TransactionsModal";
import {
  Account,
  Transaction,
  TabType,
  NativeChartConfig,
  BankChartData,
  ToolParameters,
} from "@/types";
import { buildChartConfig } from "./_utils/ChartBuilder";
import { SidebarLeft } from "./_components/SidebarLeft";
import { SidebarRight } from "./_components/SidebarRight";
import { QuickActionPanel } from "./_components/QuickActions";
import { ChartDisplay } from "./_components/ChartDisplay";
import { TransactionsControl } from "./_components/TransasctionsControl";

interface DashboardState {
  accounts: Account[];
  activeAccountId: string | null;
  selectedTransactions: Transaction[];
  isModalOpen: boolean;
  chartConfig: NativeChartConfig | null;
  userName: string;
}

export default function Dashboard() {
  const router = useRouter();
  const [state, setState] = useState<DashboardState>({
    accounts: [],
    activeAccountId: null,
    selectedTransactions: [],
    isModalOpen: false,
    chartConfig: null,
    userName: "User",
  });

  const [revokingProviderId, setRevokingProviderId] = useState<string | null>(
    null,
  );
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const fetchAccounts = async (token: string): Promise<Account[]> => {
    try {
      const res = await fetch("http://localhost:8080/api/accounts/", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = (await res.json()) as { accounts?: Account[] };
      return data.accounts || [];
    } catch (error) {
      console.log(error);
      return [];
    }
  };

  useEffect(() => {
    const token = localStorage.getItem("budai_token");
    if (!token) {
      router.push("/");
      return;
    }

    (async () => {
      const fetchedAccounts = await fetchAccounts(token);
      setState((prev) => ({
        ...prev,
        accounts: fetchedAccounts,
        activeAccountId: "ALL",
      }));
    })();
  }, [router]);

  const handleRevokeAccess = async (
    providerId: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    const confirmRevoke = window.confirm(
      "Are you sure you want to disconnect this bank? All associated sub-accounts, financial data, and charts will be permanently deleted from BudAI.",
    );
    if (!confirmRevoke) return;

    setRevokingProviderId(providerId);
    const token = localStorage.getItem("budai_token") || "";

    try {
      const res = await fetch(
        `http://localhost:8080/api/accounts/${providerId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        },
      );

      if (res.ok) {
        window.location.reload();
      } else {
        const data = (await res.json()) as { error?: string };
        alert(`Failed to disconnect: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      console.log(err);
      alert("A network error occurred while trying to disconnect.");
    } finally {
      setRevokingProviderId(null);
    }
  };

  const openAccountLedger = async (account: Account) => {
    const token = localStorage.getItem("budai_token") || "";
    setState((prev) => ({ ...prev, activeAccountId: account.account_id }));

    try {
      const res = await fetch(
        `http://localhost:8080/api/accounts/${account.account_id}/transactions`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      const data = (await res.json()) as { transactions?: Transaction[] };

      setState((prev) => ({
        ...prev,
        selectedTransactions: data.transactions || [],
        isModalOpen: true,
      }));
    } catch (error) {
      console.log(error);
      alert("Failed to load transactions for this account.");
    }
  };

  const handleFetchLedger = async (fromDate: string, toDate: string) => {
    if (!state.activeAccountId || state.activeAccountId === "ALL") return;

    const token = localStorage.getItem("budai_token") || "";
    try {
      const res = await fetch(
        `http://localhost:8080/api/accounts/${state.activeAccountId}/transactions?from=${fromDate}&to=${toDate}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      const data = (await res.json()) as { transactions?: Transaction[] };

      setState((prev) => ({
        ...prev,
        selectedTransactions: data.transactions || [],
        isModalOpen: true,
      }));
    } catch (error) {
      console.log(error);
      alert("Failed to fetch custom ledger.");
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/");
  };

  const handleLinkBank = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      const res = await fetch(
        "http://localhost:8080/api/auth/truelayer/status",
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("budai_token") || ""}`,
          },
        },
      );
      if (!res.ok) return;
      const data = (await res.json()) as { auth_url?: string };
      if (data && data.auth_url && data.auth_url !== "undefined")
        window.location.href = data.auth_url;
    } catch (err) {
      console.log(err);
    }
  };

  const handleAiChartTrigger = async (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
    extraParam?: string,
  ) => {
    setIsGenerating(true);
    let targetId = aiTargetId || state.activeAccountId;
    let isCacheToken = false;

    if (aiTargetId && aiTargetId.startsWith("CACHE_")) {
      targetId = aiTargetId;
      isCacheToken = true;
    } else if (aiTargetId && aiTargetId !== "ALL") {
      const targetBanks = aiTargetId.split(",").map((b) => b.trim());
      const resolvedIds: string[] = [];

      targetBanks.forEach((bank) => {
        const matchedAccount = state.accounts.find(
          (a) =>
            a.account_id === bank ||
            a.bank_name?.toLowerCase() === bank.toLowerCase() ||
            a.provider_name?.toLowerCase() === bank.toLowerCase(),
        );
        if (matchedAccount) {
          resolvedIds.push(matchedAccount.account_id);
        } else {
          resolvedIds.push(bank);
        }
      });
      targetId = resolvedIds.join(",");
    }

    if (!targetId) {
      setIsGenerating(false);
      return;
    }

    if (
      !isCacheToken &&
      targetId !== state.activeAccountId &&
      !targetId.includes(",")
    ) {
      setState((prev) => ({ ...prev, activeAccountId: targetId }));
    }

    const token = localStorage.getItem("budai_token") || "";
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };

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
      const res = await fetch("http://localhost:8080/api/media/execute", {
        method: "POST",
        headers,
        body: JSON.stringify({
          tool_name: toolName,
          parameters: params,
        }),
      });

      if (!res.ok) throw new Error("Chart generation failed");

      const jsonRes = (await res.json()) as { data: BankChartData[] };
      const payloadData = jsonRes.data || [];

      const newConfig = buildChartConfig(
        type,
        payloadData,
        params,
        customTitle,
      );

      if (newConfig) {
        setState((prev) => ({
          ...prev,
          chartConfig: newConfig,
        }));
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const totalBalance = state.accounts.reduce(
    (sum, acc) => sum + (acc.balance ?? acc.account_balance ?? 0),
    0,
  );

  return (
    <div className="h-screen bg-[#0D1117] text-white flex overflow-hidden">
      <TransactionModal
        isOpen={state.isModalOpen}
        onClose={() => setState((prev) => ({ ...prev, isModalOpen: false }))}
        transactions={state.selectedTransactions}
        bankName={
          state.accounts.find((a) => a.account_id === state.activeAccountId)
            ?.provider_name || "Selected Account"
        }
      />

      <SidebarLeft
        accounts={state.accounts}
        activeAccountId={state.activeAccountId}
        totalBalance={totalBalance}
        userName={state.userName}
        revokingProviderId={revokingProviderId}
        onLogout={handleLogout}
        onSetActiveAccount={(id) =>
          setState((prev) => ({
            ...prev,
            activeAccountId: id,
            isModalOpen: false,
          }))
        }
        onOpenLedger={openAccountLedger}
        onRevokeAccess={handleRevokeAccess}
        onLinkBank={handleLinkBank}
      />

      <main className="flex-1 flex flex-col relative bg-[#0D1117] p-8 pb-0">
        <div className="flex-1 mb-8 flex flex-col overflow-hidden rounded-3xl border border-slate-800 bg-[#161B22] shadow-xl relative">
          <TransactionsControl
            activeAccountId={state.activeAccountId}
            onFetchTransactions={handleFetchLedger}
          />

          <QuickActionPanel
            isGenerating={isGenerating}
            activeAccountId={state.activeAccountId}
            onTriggerChart={handleAiChartTrigger}
          />

          <ChartDisplay
            isGenerating={isGenerating}
            chartConfig={state.chartConfig}
          />
        </div>

        <div className="h-[45vh] shrink-0 pb-8">
          <BudAIChat
            onAiAction={handleAiChartTrigger}
            activeAccountId={state.activeAccountId}
            accounts={state.accounts}
          />
        </div>
      </main>

      <SidebarRight />
    </div>
  );
}
