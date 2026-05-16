"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  LayoutDashboard,
  ArrowRightLeft,
  Clock,
  RefreshCcw,
  CreditCard,
  TrendingUp,
  Sparkles,
  LineChart,
  Target,
  ShieldCheck,
} from "lucide-react";
import {
  Button,
  Link,
  Avatar,
  Card,
  Select,
  ListBox,
  Skeleton,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { BankChartData } from "@/types";
import SimulationControlsModal, {
  SimulationOverrides,
} from "@/app/(protected)/_components/SimulationControlsModal";

export default function ForecastingPage() {
  const { userName, accounts } = useBudAI();
  const [isLoading, setIsLoading] = useState(false);
  const [wealthForecast, setWealthForecast] = useState<BankChartData[] | null>(
    null,
  );
  const [expenseForecast, setExpenseForecast] = useState<
    BankChartData[] | null
  >(null);
  const [localAccountId, setLocalAccountId] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [simulationOverrides, setSimulationOverrides] =
    useState<SimulationOverrides>({
      discipline_multiplier: 1.0,
      drift_adjustment: 0.0,
      macro_environment: "Stable",
      stress_test_active: false,
      days: 60,
    });

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  const fetchForecast = useCallback(async () => {
    if (!localAccountId) return;
    setIsLoading(true);
    try {
      const [wealthRes, expenseRes] = await Promise.all([
        apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "generate_financial_forecast",
              parameters: {
                bank_name_or_id: localAccountId,
                ...simulationOverrides,
              },
            }),
          },
          true,
        ),
        apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "generate_expense_forecast",
              parameters: {
                bank_name_or_id: localAccountId,
                ...simulationOverrides,
              },
            }),
          },
          true,
        ),
      ]);

      if (wealthRes.ok) {
        const result = await wealthRes.json();
        setWealthForecast(
          Array.isArray(result.data) ? result.data : [result.data],
        );
      }
      if (expenseRes.ok) {
        const result = await expenseRes.json();
        setExpenseForecast(
          Array.isArray(result.data) ? result.data : [result.data],
        );
      }
    } catch (error) {
      console.error("Forecast Fetch Error:", error);
    } finally {
      setIsLoading(false);
    }
  }, [localAccountId, simulationOverrides]);

  useEffect(() => {
    fetchForecast();
  }, [fetchForecast]);

  const wealthChartConfig = useMemo(() => {
    if (!wealthForecast) return null;
    return buildChartConfig(
      "balance_forecast",
      wealthForecast,
      { bank_name_or_id: localAccountId, days: 60 },
      "Projected Wealth Trajectory (60 Days)",
    );
  }, [wealthForecast, localAccountId]);

  const expenseChartConfig = useMemo(() => {
    if (!expenseForecast) return null;
    return buildChartConfig(
      "expense_forecast",
      expenseForecast,
      { bank_name_or_id: localAccountId, days: 30 },
      "AI Expense Convergence (30 Days)",
    );
  }, [expenseForecast, localAccountId]);

  const stats = useMemo(() => {
    if (!wealthForecast?.[0]?.data?.length)
      return { projected: 0, change: 0, current: 0 };
    const data = wealthForecast[0].data;
    const current = Number(data[0]?.["Expected Balance"] || 0);
    const projected = Number(data[data.length - 1]?.["Expected Balance"] || 0);
    const change = current !== 0 ? ((projected - current) / current) * 100 : 0;
    return { projected, change, current };
  }, [wealthForecast]);

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/5 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/5 blur-[180px]"></div>
      </div>

      <div className="relative z-10 w-64 h-full obsidian-glass flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-neon-cyan to-[#0088FF] flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.4)]">
              <span className="text-obsidian font-black text-lg leading-none tracking-tighter">
                B
              </span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">
              BudAI
            </h1>
          </div>
          <nav className="space-y-2">
            <Link
              href="/home"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <LayoutDashboard size={20} />
              <span className="font-medium text-sm">Dashboard</span>
            </Link>
            <Link
              href="/transactions"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <ArrowRightLeft size={20} />
              <span className="font-medium text-sm">Transactions</span>
            </Link>
            <Link
              href="/forecasting"
              className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20"
            >
              <LineChart size={20} />
              <span className="font-semibold text-sm">Forecasting</span>
            </Link>
            <Link
              href="/health"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Health Radar</span>
            </Link>
            <Link
              href="/connections"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <CreditCard size={20} />
              <span className="font-medium text-sm">Connections</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/8 mt-8">
          <Avatar className="w-10 h-10 bg-linear-to-br from-neon-cyan to-deep-pink shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10 text-white font-bold" />
          <div className="overflow-hidden">
            <p
              suppressHydrationWarning
              className="text-white text-sm font-semibold truncate"
            >
              {userName || "User"}
            </p>
            <p className="text-neon-cyan/70 font-medium text-xs truncate tracking-wide">
              BudAI Member
            </p>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div>
            <h2 className="text-white text-3xl font-bold tracking-tight">
              Intelligence Forecasting
            </h2>
            <p className="text-[#8B8E98] text-sm mt-1">
              Phase 5 Hybrid LSTM-Bates Simulation Engine
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Button
              onPress={fetchForecast}
              className="bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30 hover:bg-neon-cyan/20 rounded-xl flex items-center gap-2 font-bold px-6 cursor-pointer"
            >
              <Sparkles size={18} />
              Sync Engine
            </Button>
          </div>
        </div>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-8 overflow-y-auto scrollbar-hide">
          <div className="lg:col-span-8 space-y-6">
            <Card className="obsidian-glass rounded-3xl p-6 min-h-100 flex flex-col shadow-2xl border border-white/5">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-neon-cyan/10 flex items-center justify-center text-neon-cyan">
                    <TrendingUp size={20} />
                  </div>
                  <div>
                    <h3 className="text-white font-bold text-lg">
                      Growth Trajectory
                    </h3>
                    <p className="text-[#8B8E98] text-xs">
                      LSTM-calibrated Bates model simulation
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex-1 w-full flex items-center justify-center">
                {isLoading ? (
                  <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
                ) : wealthChartConfig ? (
                  <CoreChartEngine config={wealthChartConfig} />
                ) : (
                  <div className="text-[#5E6272] text-center">
                    <Clock size={48} className="mx-auto mb-4 opacity-20" />
                    <p>Insufficient wealth data for Phase 5 projection</p>
                  </div>
                )}
              </div>
            </Card>

            <Card className="obsidian-glass rounded-3xl p-6 min-h-100 flex flex-col shadow-2xl border border-white/5">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-deep-pink/10 flex items-center justify-center text-deep-pink">
                    <LineChart size={20} />
                  </div>
                  <div>
                    <h3 className="text-white font-bold text-lg">
                      Expense Convergence
                    </h3>
                    <p className="text-[#8B8E98] text-xs">
                      Monte Carlo burn-rate simulation (30 Days)
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex-1 w-full flex items-center justify-center">
                {isLoading ? (
                  <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
                ) : expenseChartConfig ? (
                  <CoreChartEngine config={expenseChartConfig} />
                ) : (
                  <div className="text-[#5E6272] text-center">
                    <Clock size={48} className="mx-auto mb-4 opacity-20" />
                    <p>Insufficient spending data for burn-rate simulation</p>
                  </div>
                )}
              </div>
            </Card>
          </div>

          <div className="lg:col-span-4 space-y-6">
            <Card className="obsidian-glass rounded-3xl p-6 shadow-2xl border border-white/5">
              <h4 className="text-white font-bold mb-4 flex items-center gap-2">
                <ShieldCheck size={18} className="text-neon-cyan" />
                AI Analysis Snapshot
              </h4>
              <div className="space-y-4">
                <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                  <p className="text-xs text-[#8B8E98] uppercase tracking-wider font-bold mb-1">
                    Projected 60-Day Net Worth
                  </p>
                  {isLoading ? (
                    <div className="space-y-2 py-1">
                      <Skeleton className="h-8 w-32 rounded-lg bg-white/5" />
                      <Skeleton className="h-3 w-40 rounded bg-white/5" />
                    </div>
                  ) : (
                    <>
                      <p className="text-2xl font-bold text-neon-cyan">
                        £
                        {stats.projected.toLocaleString(undefined, {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        })}
                      </p>
                      <p
                        className={`text-xs mt-1 flex items-center gap-1 font-semibold ${stats.change >= 0 ? "text-brand-green" : "text-deep-pink"}`}
                      >
                        {stats.change >= 0 ? (
                          <TrendingUp size={12} />
                        ) : (
                          <LineChart size={12} />
                        )}
                        {stats.change >= 0 ? "+" : ""}
                        {stats.change.toFixed(2)}% from current
                      </p>
                    </>
                  )}
                </div>
                <div className="p-4 rounded-2xl bg-neon-cyan/5 border border-neon-cyan/20">
                  {isLoading ? (
                    <div className="space-y-2">
                      <Skeleton className="h-3 w-full rounded bg-white/5" />
                      <Skeleton className="h-3 w-5/6 rounded bg-white/5" />
                      <Skeleton className="h-3 w-4/6 rounded bg-white/5" />
                    </div>
                  ) : (
                    <p className="text-sm text-white font-medium leading-relaxed">
                      {stats.change > 0
                        ? "Phase 5 models detect a positive accumulation trend. Your asset velocity is outperforming baseline historical drift."
                        : "The engine suggests a slight contraction. Consider optimizing your subsistence floor to preserve liquid runway."}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            <Card className="obsidian-glass rounded-3xl p-6 shadow-2xl border border-white/5">
              <h4 className="text-white font-bold mb-4 flex items-center gap-2">
                <Target size={18} className="text-[#0088FF]" />
                Forecast Controls
              </h4>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs text-[#8B8E98] font-bold uppercase tracking-wider">
                    Active Analysis Node
                  </label>
                  <Select
                    aria-label="Select Account"
                    placeholder="Choose Account"
                    value={localAccountId}
                    onChange={(key) => setLocalAccountId(key as string)}
                    className="w-full"
                  >
                    <Select.Trigger className="bg-[#181A20] border-white/10 hover:border-white/20 h-12 rounded-xl px-4 flex justify-between items-center w-full">
                      <Select.Value className="text-white text-sm font-medium" />
                      <Select.Indicator />
                    </Select.Trigger>
                    <Select.Popover className="bg-[#181A20] border border-white/10 rounded-xl">
                      <ListBox className="p-1">
                        {accounts.map((acc) => (
                          <ListBox.Item
                            key={acc.account_id}
                            id={acc.account_id}
                            textValue={acc.bank_name}
                            className="flex flex-col px-3 py-2 rounded-lg hover:bg-white/5 cursor-pointer outline-none"
                          >
                            <span className="text-white font-medium text-sm">
                              {acc.bank_name}
                            </span>
                          </ListBox.Item>
                        ))}
                      </ListBox>
                    </Select.Popover>
                  </Select>
                </div>
                <Button
                  onPress={() => setIsModalOpen(true)}
                  className="w-full bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl h-12 font-medium cursor-pointer transition-all"
                >
                  Adjust Simulation Parameters
                </Button>
                <p className="text-[10px] text-[#5E6272] text-center leading-tight">
                  Simulations utilize 1,000 parallel paths via C++ Hybrid
                  Engine. LSTM Weights synchronized from local model state.
                </p>
              </div>
            </Card>
          </div>
        </div>
      </div>

      <SimulationControlsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onApply={setSimulationOverrides}
        initialValues={simulationOverrides}
      />
    </div>
  );
}
