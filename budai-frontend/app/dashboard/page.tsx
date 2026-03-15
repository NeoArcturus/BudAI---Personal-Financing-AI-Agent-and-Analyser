"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  CreditCard,
  Globe,
  LogOut,
  Plus,
  BarChart3,
  AlertTriangle,
  Unlink,
  Loader2,
} from "lucide-react";
import BudAIChat from "./_components/BudAIChat";
import TransactionModal from "./_components/TransactionsModal";
import DynamicChart from "./_components/DynamicChart";
import { Account, Transaction, TabType, NativeChartConfig } from "@/types";
import { ChartDataset } from "chart.js";

interface ChartDataRow {
  date?: string;
  Date?: string;
  amount?: number | string;
  Amount?: number | string;
  Category?: string;
  Total_Amount?: number | string;
  [key: string]: string | number | undefined;
}

interface ChartJsonResponse {
  data: ChartDataRow[];
}

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

  const [availableCharts, setAvailableCharts] = useState<string[]>([]);
  const [isCheckingCharts, setIsCheckingCharts] = useState(false);
  const [revokingProviderId, setRevokingProviderId] = useState<string | null>(
    null,
  );

  const fetchAccounts = async (token: string): Promise<Account[]> => {
    try {
      const res = await fetch("http://localhost:8080/api/accounts/", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = (await res.json()) as { accounts?: Account[] };
      return data.accounts || [];
    } catch (error) {
      console.error("Error fetching accounts:", error);
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

  useEffect(() => {
    if (!state.activeAccountId) return;

    const checkAvailableCharts = async () => {
      setIsCheckingCharts(true);
      const token = localStorage.getItem("budai_token") || "";
      const targetId = state.activeAccountId;

      const baseCharts = [
        { id: "historical_daily", prefix: "daily_spend" },
        { id: "historical_weekly", prefix: "weekly_spend" },
        { id: "historical_monthly", prefix: "monthly_spend" },
        { id: "categorized", prefix: "total_per_category" },
        { id: "expense_forecast", prefix: "converged_expense" },
        { id: "balance_forecast", prefix: "hybrid_paths" },
      ];

      const results = await Promise.all(
        baseCharts.map(async (chart) => {
          try {
            const res = await fetch(
              `http://localhost:8080/api/media/csv/${chart.prefix}_${targetId}.csv`,
              {
                method: "HEAD",
                headers: { Authorization: `Bearer ${token}` },
              },
            );

            return res.ok ? chart.id : null;
          } catch (error) {
            console.error("Error:", error);
            return null;
          }
        }),
      );

      setAvailableCharts(results.filter(Boolean) as string[]);
      setIsCheckingCharts(false);
    };

    checkAvailableCharts();
  }, [state.activeAccountId]);

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
        const data = await res.json();
        alert(`Failed to disconnect: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Error occured:", err);
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
      console.error("Error occured:", error);
      alert("Failed to load transactions for this account.");
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/");
  };

  const getChartDisplayName = (chartId: string) => {
    switch (chartId) {
      case "historical_daily":
        return "Daily Spend";
      case "historical_weekly":
        return "Weekly Spend";
      case "historical_monthly":
        return "Monthly Spend";
      case "categorized":
        return "Category Breakdown";
      case "expense_forecast":
        return "Expense Forecast";
      case "balance_forecast":
        return "Balance Projection";
      default:
        return "View Chart";
    }
  };

  // --- UPDATED: Now accepts an optional aiTargetId from BudAIChat ---
  const handleAiChartTrigger = async (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
  ) => {
    let targetId = aiTargetId || state.activeAccountId;

    // Resolve bank names (like "Wise") to the actual alphanumeric account_id
    if (aiTargetId && aiTargetId !== "ALL") {
      const matchedAccount = state.accounts.find(
        (a) =>
          a.account_id === aiTargetId ||
          a.bank_name?.toLowerCase() === aiTargetId.toLowerCase() ||
          a.provider_name?.toLowerCase() === aiTargetId.toLowerCase(),
      );
      if (matchedAccount) {
        targetId = matchedAccount.account_id;
      }
    }

    if (!targetId) return;

    // Automatically switch the UI sidebar to the account BudAI is analyzing
    if (targetId !== state.activeAccountId) {
      setState((prev) => ({ ...prev, activeAccountId: targetId }));
    }

    const token = localStorage.getItem("budai_token") || "";
    const headers = { Authorization: `Bearer ${token}` };

    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1000,
        easing: "easeOutQuart" as const,
      },
      plugins: {
        legend: {
          position: "top" as const,
          labels: { color: "#94a3b8", font: { family: "monospace" } },
        },
      },
      scales: {
        y: {
          grace: "5%",
          grid: { color: "#1e293b" },
          ticks: { color: "#94a3b8", font: { family: "monospace" } },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8", maxTicksLimit: 12, maxRotation: 0 },
        },
      },
    };

    const colorPalette = [
      "#00FFAA",
      "#3b82f6",
      "#ef4444",
      "#f59e0b",
      "#a855f7",
      "#ec4899",
    ];

    try {
      if (type === "categorized") {
        const res = await fetch(
          `http://localhost:8080/api/media/csv/total_per_category_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const json = (await res.json()) as ChartJsonResponse;
        const labels = json.data.map((d) => String(d.Category || "Unknown"));
        const amounts = json.data.map((d) => Number(d.Total_Amount || 0));

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "bar",
            data: {
              labels,
              datasets: [
                {
                  label: "Total Spent (£)",
                  data: amounts,
                  backgroundColor: "#00FFAA",
                  borderRadius: 4,
                },
              ],
            },
            options: {
              ...baseOptions,
              scales: {
                ...baseOptions.scales,
                y: { ...baseOptions.scales.y, beginAtZero: true },
              },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || "Spending Breakdown by Category",
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as NativeChartConfig,
        }));
      } else if (type === "categorized_doughnut") {
        const res = await fetch(
          `http://localhost:8080/api/media/csv/total_per_category_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const json = (await res.json()) as ChartJsonResponse;
        const filteredData = json.data.filter(
          (d) => String(d.Category).toLowerCase() !== "income",
        );
        const labels = filteredData.map((d) => String(d.Category || "Unknown"));
        const amounts = filteredData.map((d) => Number(d.Total_Amount || 0));

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "doughnut",
            data: {
              labels,
              datasets: [
                {
                  data: amounts,
                  backgroundColor: colorPalette,
                  borderColor: "#161B22",
                  borderWidth: 4,
                  hoverOffset: 10,
                },
              ],
            },
            options: {
              ...baseOptions,
              cutout: "75%",
              scales: { x: { display: false }, y: { display: false } },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || "Expense Distribution",
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as NativeChartConfig,
        }));
      } else if (type === "cash_flow_mixed") {
        const res = await fetch(
          `http://localhost:8080/api/media/csv/cash_flow_mixed_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const json = (await res.json()) as ChartJsonResponse;
        const labels = json.data.map((d) => String(d.Month));
        const income = json.data.map((d) => Number(d.Income));
        const expense = json.data.map((d) => Number(d.Expense));
        const netBalance = json.data.map((d) => Number(d.Net_Balance));

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "bar",
            data: {
              labels,
              datasets: [
                {
                  type: "line",
                  label: "Net Flow (£)",
                  data: netBalance,
                  borderColor: "#3b82f6",
                  backgroundColor: "rgba(59, 130, 246, 0.1)",
                  borderWidth: 3,
                  tension: 0.4,
                  fill: true,
                  yAxisID: "y",
                },
                {
                  type: "bar",
                  label: "Income (£)",
                  data: income,
                  backgroundColor: "#00FFAA",
                  borderRadius: 4,
                  yAxisID: "y",
                },
                {
                  type: "bar",
                  label: "Expenses (£)",
                  data: expense,
                  backgroundColor: "#ef4444",
                  borderRadius: 4,
                  yAxisID: "y",
                },
              ],
            },
            options: {
              ...baseOptions,
              scales: {
                ...baseOptions.scales,
                y: { ...baseOptions.scales.y, beginAtZero: true },
              },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || "Income vs Expense Matrix",
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as NativeChartConfig,
        }));
      } else if (type === "health_radar") {
        const res = await fetch(
          `http://localhost:8080/api/media/csv/health_radar_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const json = (await res.json()) as ChartJsonResponse;
        const labels = json.data.map((d) => String(d.Metric));
        const scores = json.data.map((d) => Number(d.Score));

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "radar",
            data: {
              labels,
              datasets: [
                {
                  label: "Health Index",
                  data: scores,
                  backgroundColor: "rgba(0, 255, 170, 0.2)",
                  borderColor: "#00FFAA",
                  borderWidth: 2,
                  pointBackgroundColor: "#161B22",
                  pointBorderColor: "#00FFAA",
                  pointHoverBackgroundColor: "#fff",
                  pointHoverBorderColor: "#00FFAA",
                },
              ],
            },
            options: {
              ...baseOptions,
              scales: {
                x: { display: false },
                y: { display: false },
                r: {
                  angleLines: { color: "#1e293b" },
                  grid: { color: "#1e293b" },
                  pointLabels: {
                    color: "#94a3b8",
                    font: { family: "monospace", size: 11 },
                  },
                  ticks: { display: false, min: 0, max: 100, stepSize: 20 },
                },
              },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || "Financial Health Profile",
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as NativeChartConfig,
        }));
      } else if (type === "expense_forecast" || type === "balance_forecast") {
        const prefix =
          type === "expense_forecast" ? "converged_expense" : "hybrid_paths";

        const res = await fetch(
          `http://localhost:8080/api/media/csv/${prefix}_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const rawText = await res.text();
        const rows = rawText
          .trim()
          .split(/\r?\n/)
          .filter((r) => r !== "");

        const datasets: ChartDataset<"line">[] = [];
        let labels: string[] = [];

        const pathConfigs = [
          {
            border: "#00FFAA",
            bg: "rgba(0, 255, 170, 0.1)",
            label: "Expected Balance",
          },
          {
            border: "#ef4444",
            bg: "rgba(239, 68, 68, 0.1)",
            label: "Careless Scenario (5%)",
          },
          {
            border: "#3b82f6",
            bg: "rgba(59, 130, 246, 0.1)",
            label: "Optimal Scenario (95%)",
          },
        ];

        rows.forEach((row, rIdx) => {
          const parts = row
            .split(",")
            .map((v) => v.trim())
            .filter((v) => v !== "");
          if (parts.length === 0) return;

          let labelName = "Expected Balance";
          let borderColor = pathConfigs[0].border;
          let bgColor = pathConfigs[0].bg;

          if (isNaN(Number(parts[0]))) {
            labelName = parts[0];
            parts.shift();
            borderColor = colorPalette[rIdx % colorPalette.length];
            bgColor = "transparent";
          } else {
            const config =
              type === "balance_forecast" && rows.length === 3
                ? pathConfigs[rIdx]
                : pathConfigs[0];
            labelName =
              type === "expense_forecast"
                ? "Projected Daily Spend (£)"
                : config.label;
            borderColor =
              type === "expense_forecast" ? "#ef4444" : config.border;
            bgColor =
              type === "expense_forecast"
                ? "rgba(239, 68, 68, 0.1)"
                : config.bg;
          }

          const parsedAmounts = parts
            .map((v) => Number(v))
            .filter((n) => !isNaN(n));
          if (rIdx === 0) labels = parsedAmounts.map((_, i) => `Day ${i}`);

          datasets.push({
            label: labelName,
            data: parsedAmounts,
            borderColor: borderColor,
            backgroundColor: bgColor,
            fill:
              datasets.length === 0 &&
              rows.length === 1 &&
              !isNaN(Number(row.split(",")[0]))
                ? true
                : false,
            tension: 0.4,
            pointRadius: 0,
            pointHitRadius: 10,
          });
        });

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "line",
            data: { labels, datasets },
            options: {
              ...baseOptions,
              scales: {
                ...baseOptions.scales,
                y: {
                  ...baseOptions.scales.y,
                  beginAtZero: type === "expense_forecast",
                },
              },
              interaction: { mode: "index", intersect: false },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || "AI Forecast Analysis",
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as unknown as NativeChartConfig,
        }));
      } else if (type.startsWith("historical")) {
        const timeType = type.includes("_") ? type.split("_")[1] : "monthly";

        const res = await fetch(
          `http://localhost:8080/api/media/csv/${timeType}_spend_${targetId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        const json = (await res.json()) as ChartJsonResponse;

        const labels = json.data.map((d) => {
          const rawDate = String(d.date || d.Date || "Unknown");
          return rawDate.split(" ")[0].split("T")[0];
        });

        const datasets: ChartDataset<"line">[] = [];
        const firstRow = json.data[0] || {};
        const bankKeys = Object.keys(firstRow).filter(
          (k) => !["date", "Date", "amount", "Amount"].includes(k),
        );

        if (bankKeys.length > 0) {
          bankKeys.forEach((bank, idx) => {
            const amounts = json.data.map((d) => Number(d[bank] || 0));
            datasets.push({
              label: bank,
              data: amounts,
              borderColor: colorPalette[idx % colorPalette.length],
              fill: false,
              tension: 0.4,
              pointRadius: 0,
              pointHitRadius: 10,
            });
          });
        } else {
          const amounts = json.data.map((d) =>
            Number(d.amount || d.Amount || 0),
          );
          datasets.push({
            label: `Historical Expenses`,
            data: amounts,
            borderColor: "#3b82f6",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHitRadius: 10,
          });
        }

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "line",
            data: { labels, datasets },
            options: {
              ...baseOptions,
              scales: {
                ...baseOptions.scales,
                y: { ...baseOptions.scales.y, beginAtZero: true },
              },
              interaction: { mode: "index", intersect: false },
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text:
                    customTitle ||
                    `${timeType.charAt(0).toUpperCase() + timeType.slice(1)} Expense Analysis`,
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as unknown as NativeChartConfig,
        }));
      }
    } catch (err) {
      console.error("Error during chart generation:", err);
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

      <aside className="w-[22%] p-6 flex flex-col gap-4 overflow-y-auto shrink-0 border-r border-slate-800 scrollbar-hide">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase">
              Welcome Back
            </h2>
            <p className="text-[#00FFAA] font-bold text-lg capitalize">
              {state.userName}
            </p>
          </div>
          <button
            type="button"
            onClick={handleLogout}
            title="Logout"
            className="text-slate-500 hover:text-red-400 transition-colors bg-[#1c2128] p-2 rounded-xl border border-slate-800"
          >
            <LogOut size={16} />
          </button>
        </div>

        <div
          onClick={() =>
            setState((prev) => ({
              ...prev,
              activeAccountId: "ALL",
              isModalOpen: false,
            }))
          }
          className={`bg-[#161B22] p-5 rounded-2xl border ${
            state.activeAccountId === "ALL"
              ? "border-[#00FFAA]"
              : "border-slate-800 hover:border-[#00FFAA]/50"
          } cursor-pointer transition-all shrink-0 group relative`}
        >
          <div className="flex items-center gap-3 mb-4 pr-8">
            <Globe className="text-[#00FFAA] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
            <span className="text-sm font-bold text-slate-200">
              All Accounts
            </span>
          </div>
          <h3 className="text-2xl font-mono font-bold text-white">
            £
            {totalBalance.toLocaleString(undefined, {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </h3>
        </div>

        {state.accounts.map((acc, idx) => {
          const isActive = acc.account_id === state.activeAccountId;

          return (
            <div
              key={idx}
              onClick={() => openAccountLedger(acc)}
              className={`bg-[#161B22] p-5 rounded-2xl border ${isActive ? "border-[#00FFAA]" : "border-slate-800 hover:border-[#00FFAA]/50"} cursor-pointer transition-all shrink-0 group relative`}
            >
              <button
                onClick={(e) =>
                  handleRevokeAccess(acc.provider_id || acc.account_id, e)
                }
                disabled={revokingProviderId === acc.provider_id}
                className="absolute top-4 right-4 text-slate-600 hover:text-red-500 transition-colors bg-[#0D1117] p-1.5 rounded-lg border border-slate-800 disabled:opacity-50"
                title="Disconnect Bank Account"
              >
                {revokingProviderId === acc.provider_id ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Unlink size={14} />
                )}
              </button>

              <div className="flex items-center gap-3 mb-4 pr-8">
                <CreditCard className="text-[#00FFAA] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
                <div className="flex flex-col truncate">
                  <span className="text-sm font-bold text-slate-200 truncate">
                    {acc.provider_name || acc.bank_name}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono tracking-widest mt-0.5">
                    {acc.sort_code} | ••••{acc.account_number}
                  </span>
                </div>
              </div>
              <h3 className="text-2xl font-mono font-bold text-white">
                {acc.currency === "GBP" ? "£" : acc.currency}
                {(acc.balance ?? acc.account_balance ?? 0).toLocaleString(
                  undefined,
                  { minimumFractionDigits: 2, maximumFractionDigits: 2 },
                )}
              </h3>
              {acc.status === "revoked" && (
                <div className="mt-4 flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-400 border border-red-500/50 rounded-lg text-xs font-bold">
                  <AlertTriangle size={14} /> Reconnect Required
                </div>
              )}
            </div>
          );
        })}

        <button
          type="button"
          onClick={async (e) => {
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
              console.error("Error initiating bank linking:", err);
            }
          }}
          className="w-full mt-2 flex items-center justify-center gap-2 bg-[#161B22] border border-slate-800 text-slate-400 p-4 rounded-2xl hover:border-[#00FFAA]/50 hover:text-[#00FFAA] transition-all border-dashed"
        >
          <Plus size={18} />{" "}
          <span className="text-xs font-bold tracking-widest uppercase">
            Link Bank
          </span>
        </button>
      </aside>

      <main className="flex-1 flex flex-col relative bg-[#0D1117] p-8 pb-0">
        <div className="flex-1 mb-8 flex flex-col overflow-hidden rounded-3xl border border-slate-800 bg-[#161B22] shadow-xl relative">
          {/* Display buttons for ANY account if charts exist */}
          {availableCharts.length > 0 && (
            <div className="flex flex-wrap gap-2 p-4 border-b border-slate-800 bg-[#0D1117]/50 items-center justify-center z-10">
              {isCheckingCharts ? (
                <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
              ) : (
                availableCharts.map((chartId) => (
                  <button
                    key={chartId}
                    onClick={() => handleAiChartTrigger(chartId as TabType)}
                    className="bg-[#161B22] border border-slate-700 hover:border-[#00FFAA] text-slate-300 hover:text-[#00FFAA] transition-all px-4 py-2 rounded-xl text-xs font-bold"
                  >
                    {getChartDisplayName(chartId)}
                  </button>
                ))
              )}
            </div>
          )}

          <div className="flex-1 relative p-4 flex flex-col items-center justify-center">
            {state.chartConfig ? (
              <DynamicChart config={state.chartConfig} />
            ) : (
              <div className="text-center max-w-md">
                <BarChart3 className="w-16 h-16 mb-4 opacity-50 mx-auto text-[#00FFAA]" />
                <p className="text-sm tracking-widest uppercase font-bold text-slate-400">
                  {availableCharts.length > 0
                    ? "Select a Dashboard"
                    : "Awaiting AI Analysis"}
                </p>
                <p className="text-xs opacity-50 mt-2">
                  {availableCharts.length > 0
                    ? "Click one of the buttons above to view your available charts, or ask BudAI for a new analysis."
                    : "No data visualizations are available for this account yet. Ask BudAI to plot your monthly expenses or generate a forecast."}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="h-[45vh] shrink-0 pb-8">
          <BudAIChat
            onAiAction={handleAiChartTrigger}
            activeAccountId={state.activeAccountId}
            accounts={state.accounts}
          />
        </div>
      </main>

      <aside className="w-[22%] p-6 bg-[#161B22] border-l border-slate-800 overflow-y-auto shrink-0 scrollbar-hide">
        <h2 className="text-xs font-bold tracking-widest text-slate-500 mb-6 flex items-center gap-2">
          <Globe className="w-4 h-4" /> GLOBAL MARKETS
        </h2>
        <div className="space-y-4">
          {[1, 2, 3].map((item) => (
            <div
              key={item}
              className="bg-[#0D1117] p-5 rounded-2xl border border-slate-800"
            >
              <div className="h-4 w-1/4 bg-slate-800 rounded mb-3"></div>
              <div className="h-4 w-full bg-slate-800 rounded mb-2"></div>
              <div className="h-4 w-5/6 bg-slate-800 rounded"></div>
              <p className="text-[10px] text-slate-500 mt-4 uppercase">
                Geopolitical Event
              </p>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
