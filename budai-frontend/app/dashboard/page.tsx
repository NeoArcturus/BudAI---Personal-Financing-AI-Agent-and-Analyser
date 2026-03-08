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
} from "lucide-react";
import BudAIChat from "./_components/BudAIChat";
import TransactionModal from "./_components/TransactionsModal";
import DynamicChart from "./_components/DynamicChart";
import { Account, Transaction, TabType } from "@/types";
import {
  ChartConfiguration,
  ScriptableContext,
  ChartDataset,
  PointElement,
} from "chart.js";

// --- STRICT TYPING: Replaces 'any' for all CSV/JSON parsing ---
interface ChartDataRow {
  date?: string;
  Date?: string;
  amount?: number | string;
  Amount?: number | string;
  Category?: string;
  Total_Amount?: number | string;
}

interface ChartJsonResponse {
  data: ChartDataRow[];
}

interface DashboardState {
  accounts: Account[];
  activeAccountId: string | null;
  selectedTransactions: Transaction[];
  isModalOpen: boolean;
  chartConfig: ChartConfiguration | null;
  userName: string;
}

// Strictly typed caching context for the progressive animation to prevent browser freezing
type AnimContext = ScriptableContext<"line"> & {
  xStarted?: boolean;
  yStarted?: number;
  yStartedDelay?: boolean;
};

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

  const fetchAccounts = async (token: string) => {
    try {
      const res = await fetch("http://localhost:8080/api/accounts/", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = (await res.json()) as { accounts?: Account[] };
      return data.accounts || [];
    } catch (error) {
      console.error("Failed to fetch accounts", error);
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
      const firstAccountId =
        fetchedAccounts.length > 0
          ? fetchedAccounts[0].truelayer_account_id ||
            fetchedAccounts[0].account_id
          : null;

      setState((prev) => ({
        ...prev,
        accounts: fetchedAccounts,
        activeAccountId: firstAccountId,
      }));
    })();
  }, [router]);

  const openAccountLedger = async (account: Account) => {
    const token = localStorage.getItem("budai_token") || "";
    const accountId = account.truelayer_account_id || account.account_id;

    setState((prev) => ({ ...prev, activeAccountId: accountId }));

    try {
      const res = await fetch(
        `http://localhost:8080/api/accounts/${accountId}/transactions`,
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
      console.error("Failed to fetch transactions", error);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/");
  };

  const handleAiChartTrigger = async (
    type: TabType | "historical",
    customTitle?: string,
  ) => {
    if (!state.activeAccountId) return;
    const token = localStorage.getItem("budai_token") || "";
    const headers = { Authorization: `Bearer ${token}` };

    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top" as const,
          labels: { color: "#94a3b8", font: { family: "monospace" } },
        },
      },
      scales: {
        y: {
          grid: { color: "#1e293b" },
          ticks: { color: "#94a3b8", font: { family: "monospace" } },
        },
        x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
      },
    };

    try {
      if (type === "categorized") {
        const res = await fetch(
          `http://localhost:8080/api/media/csv/total_per_category_${state.activeAccountId}.csv`,
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
              animation: {
                duration: 1000,
                delay: (ctx: ScriptableContext<"bar">) =>
                  ctx.type === "data" ? ctx.dataIndex * 100 : 0,
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
          } as ChartConfiguration,
        }));
      } else if (type === "expense_forecast" || type === "balance_forecast") {
        const prefix =
          type === "expense_forecast" ? "converged_expense" : "hybrid_paths";
        const res = await fetch(
          `http://localhost:8080/api/media/csv/${prefix}_${state.activeAccountId}.csv`,
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

        let maxPoints = 0;

        rows.forEach((row, rIdx) => {
          const parsedAmounts = row
            .split(",")
            .map((v) => v.trim())
            .filter((v) => v !== "")
            .map((v) => Number(v))
            .filter((n) => !isNaN(n));
          if (parsedAmounts.length === 0) return;

          if (parsedAmounts.length > maxPoints)
            maxPoints = parsedAmounts.length;
          if (rIdx === 0) labels = parsedAmounts.map((_, i) => `Day ${i}`);

          const config =
            type === "balance_forecast" && rows.length === 3
              ? pathConfigs[rIdx]
              : pathConfigs[0];

          datasets.push({
            label:
              type === "expense_forecast"
                ? "Projected Daily Spend (£)"
                : config.label,
            data: parsedAmounts,
            borderColor:
              type === "expense_forecast" ? "#ef4444" : config.border,
            backgroundColor:
              type === "expense_forecast"
                ? "rgba(239, 68, 68, 0.1)"
                : config.bg,
            fill: datasets.length === 0 && rows.length === 1 ? true : false,
            tension: 0.4,
            pointRadius: 0,
            pointHitRadius: 10,
          });
        });

        const totalDuration = 2500;
        const delayBetweenPoints =
          maxPoints > 0 ? totalDuration / maxPoints : 0;

        const progressiveAnimation = {
          x: {
            type: "number",
            easing: "linear",
            duration: delayBetweenPoints,
            from: NaN,
            delay: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data" || aCtx.xStarted) return 0;
              aCtx.xStarted = true;
              return aCtx.dataIndex * delayBetweenPoints;
            },
          },
          y: {
            type: "number",
            easing: "linear",
            duration: delayBetweenPoints,
            from: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data") return 0;

              if (aCtx.dataIndex === 0) {
                const datasetIndex = aCtx.datasetIndex;
                const startingValue =
                  (datasets[datasetIndex]?.data[0] as number) || 0;
                return (
                  aCtx.chart.scales.y?.getPixelForValue(startingValue) || 0
                );
              }

              if (aCtx.yStarted === undefined) {
                const meta = aCtx.chart.getDatasetMeta(aCtx.datasetIndex);
                const prev = meta.data[aCtx.dataIndex - 1] as PointElement;
                aCtx.yStarted = prev ? prev.y : 0;
              }

              return aCtx.yStarted;
            },
            delay: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data" || aCtx.yStartedDelay) return 0;
              aCtx.yStartedDelay = true;
              return aCtx.dataIndex * delayBetweenPoints;
            },
          },
        };

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "line",
            data: { labels, datasets },
            options: {
              ...baseOptions,
              interaction: {
                mode: "index",
                intersect: false,
              },
              animations: progressiveAnimation,
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
          } as unknown as ChartConfiguration,
        }));
      } else if (type.startsWith("historical")) {
        // Extract the time type (e.g., "historical_monthly" -> "monthly")
        const timeType = type.includes("_") ? type.split("_")[1] : "monthly";

        // Fetch the specific file based on the timeframe
        const res = await fetch(
          `http://localhost:8080/api/media/csv/${timeType}_spend_${state.activeAccountId}.csv`,
          { headers },
        );
        if (!res.ok) return;

        // Cast using the strict interface defined at the top
        const json = (await res.json()) as ChartJsonResponse;

        // Safely map values, formatting the date to remove the time component
        const labels = json.data.map((d) => {
          const rawDate = String(d.date || d.Date || "Unknown");
          return rawDate.split(" ")[0].split("T")[0];
        });

        const amounts = json.data.map((d) => Number(d.amount || d.Amount || 0));
        const timeTitle = timeType.charAt(0).toUpperCase() + timeType.slice(1);

        // --- PROGRESSIVE ANIMATION MATH ---
        const maxPoints = amounts.length;
        const totalDuration = 2500;
        const delayBetweenPoints =
          maxPoints > 0 ? totalDuration / maxPoints : 0;

        const progressiveAnimation = {
          x: {
            type: "number",
            easing: "linear",
            duration: delayBetweenPoints,
            from: NaN,
            delay: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data" || aCtx.xStarted) return 0;
              aCtx.xStarted = true;
              return aCtx.dataIndex * delayBetweenPoints;
            },
          },
          y: {
            type: "number",
            easing: "linear",
            duration: delayBetweenPoints,
            from: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data") return 0;

              if (aCtx.dataIndex === 0) {
                const startingValue = amounts[0] || 0;
                return (
                  aCtx.chart.scales.y?.getPixelForValue(startingValue) || 0
                );
              }

              if (aCtx.yStarted === undefined) {
                const meta = aCtx.chart.getDatasetMeta(aCtx.datasetIndex);
                const prev = meta.data[aCtx.dataIndex - 1] as PointElement;
                aCtx.yStarted = prev ? prev.y : 0;
              }

              return aCtx.yStarted;
            },
            delay: (ctx: ScriptableContext<"line">) => {
              const aCtx = ctx as AnimContext;
              if (aCtx.type !== "data" || aCtx.yStartedDelay) return 0;
              aCtx.yStartedDelay = true;
              return aCtx.dataIndex * delayBetweenPoints;
            },
          },
        };

        setState((prev) => ({
          ...prev,
          chartConfig: {
            type: "line",
            data: {
              labels,
              datasets: [
                {
                  label: `Historical Expenses (${timeTitle})`,
                  data: amounts,
                  borderColor: "#3b82f6",
                  fill: false,
                  pointRadius: 0,
                  pointHitRadius: 10,
                  pointHoverRadius: 4,
                },
              ],
            },
            options: {
              ...baseOptions,
              interaction: {
                mode: "index",
                intersect: false,
              },
              animations: progressiveAnimation, // Inject the animation here
              plugins: {
                ...baseOptions.plugins,
                title: {
                  display: true,
                  text: customTitle || `${timeTitle} Expense Analysis`,
                  color: "#ffffff",
                  font: { size: 16 },
                },
              },
            },
          } as unknown as ChartConfiguration,
        }));
      }
    } catch (err) {
      console.error("Chart parsing failed:", err);
    }
  };

  return (
    <div className="h-screen bg-[#0D1117] text-white flex overflow-hidden">
      <TransactionModal
        isOpen={state.isModalOpen}
        onClose={() => setState((prev) => ({ ...prev, isModalOpen: false }))}
        transactions={state.selectedTransactions}
        bankName={
          state.accounts.find(
            (a) =>
              (a.truelayer_account_id || a.account_id) ===
              state.activeAccountId,
          )?.bank_name || "Selected Account"
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

        {state.accounts.map((acc, idx) => (
          <div
            key={idx}
            onClick={() => openAccountLedger(acc)}
            className={`bg-[#161B22] p-5 rounded-2xl border ${(acc.truelayer_account_id || acc.account_id) === state.activeAccountId ? "border-[#00FFAA]" : "border-slate-800 hover:border-[#00FFAA]/50"} cursor-pointer transition-all shrink-0 group relative`}
          >
            <div className="flex items-center gap-3 mb-4">
              <CreditCard className="text-[#00FFAA] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
              <div className="flex flex-col truncate">
                <span className="text-sm font-bold text-slate-200 truncate">
                  {acc.bank_name || acc.provider_name}
                </span>
                <span className="text-[10px] text-slate-500 font-mono tracking-widest mt-0.5">
                  {acc.sort_code} | ••••{acc.account_number}
                </span>
              </div>
            </div>
            <h3 className="text-2xl font-mono font-bold text-white">
              {acc.currency === "GBP" ? "£" : acc.currency}
              {(acc.balance ?? acc.account_balance ?? 0).toLocaleString()}
            </h3>
            {acc.status === "revoked" && (
              <div className="mt-4 flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-400 border border-red-500/50 rounded-lg text-xs font-bold">
                <AlertTriangle size={14} /> Reconnect Required
              </div>
            )}
          </div>
        ))}

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
              console.error("Network error fetching bank link:", err);
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
        <div className="flex-1 mb-8 overflow-hidden rounded-3xl border border-slate-800 bg-[#161B22] shadow-xl">
          {state.chartConfig ? (
            <DynamicChart config={state.chartConfig} />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-600 bg-[#0D1117]">
              <BarChart3 className="w-16 h-16 mb-4 opacity-50" />
              <p className="text-sm tracking-widest uppercase font-bold text-slate-400">
                Awaiting AI Analysis
              </p>
              <p className="text-xs opacity-50 mt-2">
                Talk with BudAI to generate insights, forecasts, and
                visualizations based on your financial data. Try asking for a
                spending breakdown or a 30-day forecast!
              </p>
            </div>
          )}
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
