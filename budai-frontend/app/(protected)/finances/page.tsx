"use client";

import React, { useState, useEffect } from "react";
import { useBudAI } from "@/app/context/AppContext";
import { QuickActionPanel } from "@/app/(protected)/_components/QuickActions";
import { ChartDisplay } from "@/app/(protected)/_components/ChartDisplay";
import { TransactionsControl } from "@/app/(protected)/_components/TransactionsControl";
import TransactionModal from "@/app/(protected)/_components/TransactionsModal";
import { Transaction, NativeChartConfig } from "@/types";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { Wallet, Receipt, AlertTriangle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { apiFetch } from "@/lib/api";

export default function FinancesPage() {
  const {
    activeAccountId,
    chartConfig,
    isGenerating,
    handleAiChartTrigger,
    accounts,
    totalBalance,
  } = useBudAI();
  const [selectedTransactions, setSelectedTransactions] = useState<
    Transaction[]
  >([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [initialChartConfig, setInitialChartConfig] =
    useState<NativeChartConfig | null>(null);
  const [scaMessage, setScaMessage] = useState<string | null>(null);
  const [isFetchingInitial, setIsFetchingInitial] = useState(true);

  useEffect(() => {
    const loadInitialChart = async () => {
      try {
        const res = await apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "plot_expenses",
              parameters: {
                bank_name_or_id: "ALL",
                plot_time_type: "monthly",
                from_date: "2025-01-01",
                to_date: "2025-12-31",
              },
            }),
          },
          true,
        );
        const data = await res.json();
        if (!res.ok) {
          if (data.error && data.error.includes("SCA")) {
            setScaMessage(data.error);
          } else {
            setScaMessage(data.error || "Failed to load expenses.");
          }
        } else {
          const config = buildChartConfig(
            "historical_monthly",
            data.data || [],
            {
              bank_name_or_id: "",
            },
            "2025 Combined Expenses",
          );
          if (config) setInitialChartConfig(config);
        }
      } catch (err) {
        console.log(err);
        setScaMessage("Network error while communicating with backend.");
      } finally {
        setIsFetchingInitial(false);
      }
    };
    loadInitialChart();
  }, []);

  const handleFetchLedger = async (fromDate: string, toDate: string) => {
    if (!activeAccountId || activeAccountId === "ALL") return;
    try {
      const res = await apiFetch(
        `/api/accounts/${activeAccountId}/transactions?from=${fromDate}&to=${toDate}`,
        {},
        true,
      );
      const data = await res.json();
      setSelectedTransactions(data.transactions || []);
      setIsModalOpen(true);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="flex flex-col h-full w-full overflow-y-auto scrollbar-hide">
      <TransactionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        transactions={selectedTransactions}
        bankName={
          accounts.find((a) => a.account_id === activeAccountId)
            ?.provider_name || "Selected Account"
        }
      />

      {/* Top Header Section */}
      <div className="shrink-0 w-full p-6 pb-2">
        <div className="grid grid-cols-2 gap-6 mb-6">
          <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col justify-center">
            <span className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-2">
              <Wallet size={16} className="text-[#69F0AE]" /> Total Wealth
            </span>
            <h2 className="text-4xl font-mono font-bold text-white">
              £
              {totalBalance.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </h2>
          </div>
          <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col justify-center">
            <span className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-2">
              <Receipt size={16} className="text-rose-400" /> 2025 Combined
              Expenses
            </span>
            <h2 className="text-4xl font-mono font-bold text-white">
              £0.00{" "}
              {isFetchingInitial && (
                <span className="text-sm text-slate-500 font-sans font-normal">
                  (Fetching...)
                </span>
              )}
            </h2>
          </div>
        </div>

        <TransactionsControl
          activeAccountId={activeAccountId}
          onFetchTransactions={handleFetchLedger}
        />
        <div className="mt-6 mb-6">
          <QuickActionPanel
            isGenerating={isGenerating}
            activeAccountId={activeAccountId}
            onTriggerChart={handleAiChartTrigger}
          />
        </div>
      </div>

      <div className="flex-1 w-full p-6 pt-2 flex flex-col gap-6 shrink-0 min-h-0">
        {/* Persistent 2025 Expenses Chart */}
        <div className="w-full flex flex-col h-100 shrink-0">
          <div className="flex-1 rounded-2xl border border-[#1A2D21] bg-[#132017] shadow-xl overflow-hidden relative">
            {scaMessage && !initialChartConfig ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 bg-[#0A120D] text-center">
                <AlertTriangle size={48} className="text-rose-500 mb-4" />
                <div className="prose prose-invert max-w-lg prose-p:text-slate-300 prose-a:text-[#69F0AE]">
                  <ReactMarkdown>{scaMessage}</ReactMarkdown>
                </div>
              </div>
            ) : (
              <ChartDisplay
                isGenerating={isFetchingInitial}
                chartConfig={initialChartConfig}
              />
            )}
          </div>
        </div>

        {/* AI Agent Dynamic Chart Area (Only mounts when LLM triggers a chart) */}
        {chartConfig && (
          <div className="w-full flex flex-col h-100 shrink-0 mb-8">
            <div className="flex items-center gap-2 mb-3 px-2">
              <div className="w-2 h-2 rounded-full bg-[#69F0AE] animate-pulse"></div>
              <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest">
                BudAI&apos;s Findings
              </h3>
            </div>
            <div className="flex-1 rounded-2xl border border-[#69F0AE]/30 bg-[#132017] shadow-[0_0_20px_rgba(105,240,174,0.05)] overflow-hidden relative">
              <ChartDisplay
                isGenerating={isGenerating}
                chartConfig={chartConfig}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
