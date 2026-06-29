"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Clock,
  TrendingUp,
  TrendingDown,
  Sparkles,
  LineChart,
  Target,
  ShieldCheck,
} from "lucide-react";
import { Button, Card, Select, ListBox, Skeleton } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { BankChartData } from "@/types";
import SimulationControlsModal, {
  SimulationOverrides,
} from "@/app/(protected)/_components/SimulationControlsModal";

export default function ForecastingPage() {
  const { accounts } = useBudAI();
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
        const result = await wealthRes.json() as any;
        const raw = result.data;
        const unwrapped = (raw && typeof raw === 'object' && 'series' in raw) ? raw.series : (Array.isArray(raw) ? raw : [raw]);
        setWealthForecast(unwrapped);
      }
      if (expenseRes.ok) {
        const result = await expenseRes.json() as any;
        const raw = result.data;
        const unwrapped = (raw && typeof raw === 'object' && 'series' in raw) ? raw.series : (Array.isArray(raw) ? raw : [raw]);
        setExpenseForecast(unwrapped);
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
    const current = Number(data[0]?.["Expected Balance"] || data[0]?.["Balance"] || 0);
    const projected = Number(data[data.length - 1]?.["Expected Balance"] || data[data.length - 1]?.["Balance"] || 0);
    const change = current !== 0 ? ((projected - current) / Math.abs(current)) * 100 : 0;
    return { projected, change, current };
  }, [wealthForecast]);

  return (
    <div className="relative z-10 flex-1 flex flex-col pt-10 px-10 h-full">
      <div className="flex items-center justify-between mb-10 shrink-0">
        <div>
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Financial Projections
          </h2>
          <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] mt-1.5">
            AI-Driven Financial Forecasting
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Button
            onPress={fetchForecast}
            className="bg-white/5 backdrop-blur-xl text-primary border-[0.5px] border-primary/20 hover:border-primary/50 rounded-xl flex items-center gap-3 font-black text-[10px] uppercase tracking-widest px-8 h-12 cursor-pointer transition-all shadow-inner"
          >
            <Sparkles size={16} />
            Update Forecast
          </Button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 pb-8 overflow-y-auto scrollbar-hide">
        <div className="lg:col-span-8 space-y-8">
          <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
            <div className="flex items-center justify-between mb-10">
              <div className="flex items-center gap-5">
                <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-sm">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <h3 className="text-foreground font-black text-lg tracking-widest uppercase italic m-0">
                    Account Balance
                  </h3>
                  <p className="text-foreground/30 text-[9px] font-black uppercase tracking-[0.4em] mt-1">
                    Statistical forecast
                  </p>
                </div>
              </div>
            </div>
            <div className="flex-1 w-full flex items-center justify-center min-h-80 bg-white/1 rounded-xl border-[0.5px] border-white/5 relative overflow-hidden p-8">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.02)_0%,transparent_70%)] pointer-events-none" />
              {isLoading ? (
                <Skeleton animationType="shimmer" className="w-full h-full rounded-lg bg-white/5" />
              ) : wealthChartConfig ? (
                <CoreChartEngine config={wealthChartConfig} />
              ) : (
                <div className="text-foreground/20 text-center opacity-40">
                  <Clock size={40} className="mx-auto mb-4" />
                  <p className="text-[9px] font-black uppercase tracking-[0.4em]">
                    Loading Data...
                  </p>
                </div>
              )}
            </div>
          </Card>

          <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
            <div className="flex items-center justify-between mb-10">
              <div className="flex items-center gap-5">
                <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-sm">
                  <LineChart size={20} />
                </div>
                <div>
                  <h3 className="text-foreground font-black text-lg tracking-widest uppercase italic m-0">
                    Expenditure
                  </h3>
                  <p className="text-foreground/30 text-[9px] font-black uppercase tracking-[0.4em] mt-1">
                    Statistical Forecast
                  </p>
                </div>
              </div>
            </div>
            <div className="flex-1 w-full flex items-center justify-center min-h-80 bg-white/1 rounded-xl border-[0.5px] border-white/5 relative overflow-hidden p-8">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.02)_0%,transparent_70%)] pointer-events-none" />
              {isLoading ? (
                <Skeleton animationType="shimmer" className="w-full h-full rounded-lg bg-white/5" />
              ) : expenseChartConfig ? (
                <CoreChartEngine config={expenseChartConfig} />
              ) : (
                <div className="text-foreground/20 text-center opacity-40">
                  <Clock size={40} className="mx-auto mb-4" />
                  <p className="text-[9px] font-black uppercase tracking-[0.4em]">
                    Awaiting Data...
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>

        <div className="lg:col-span-4 space-y-8">
          <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
            <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] mb-10 flex items-center gap-3 italic">
              <ShieldCheck size={18} className="text-primary shrink-0" />
              Financial Analysis
            </h4>
            <div className="space-y-8">
              <div className="p-8 rounded-2xl bg-white/5 border-[0.5px] border-white/10 backdrop-blur-xl shadow-inner">
                <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em] mb-3">
                  Projected Net amount
                </p>
                {isLoading ? (
                  <div className="space-y-4 py-1">
                    <Skeleton animationType="shimmer" className="h-12 w-32 rounded-lg bg-white/5" />
                    <Skeleton animationType="shimmer" className="h-2 w-40 rounded bg-white/5" />
                  </div>
                ) : (
                  <>
                    <p className="text-4xl font-normal text-primary tracking-tighter font-mono">
                      £
                      {stats.projected.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </p>
                    <p
                      className={`text-[12px] mt-4 flex items-center gap-2 font-black uppercase tracking-widest ${stats.change >= 0 ? "text-green-500" : "text-red-500"}`}
                    >
                      {stats.change >= 0 ? (
                        <TrendingUp size={30} />
                      ) : (
                        <TrendingDown size={30} />
                      )}
                      {stats.change >= 0 ? "+" : ""}
                      {stats.change.toFixed(2)}% rate (based on your current
                      balance)
                    </p>
                  </>
                )}
              </div>
              <div className="p-8 rounded-2xl bg-primary/5 border-[0.5px] border-primary/20 backdrop-blur-xl">
                {isLoading ? (
                  <div className="space-y-3">
                    <Skeleton animationType="shimmer" className="h-2 w-full rounded bg-white/5" />
                    <Skeleton animationType="shimmer" className="h-2 w-5/6 rounded bg-white/5" />
                  </div>
                ) : (
                  <p className="text-[11px] text-foreground/60 font-medium leading-relaxed tracking-wide uppercase">
                    {stats.change > 0
                      ? "Positive growth trend identified. Your savings rate is higher than historical averages."
                      : "Forecast suggests a decrease in balance. Consider reducing non-essential spending."}
                  </p>
                )}
              </div>
            </div>
          </Card>

          <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
            <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] mb-10 flex items-center gap-3 italic">
              <Target size={18} className="text-primary shrink-0" />
              Parameters
            </h4>
            <div className="space-y-8">
              <div className="space-y-4">
                <label className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] pl-1">
                  Bank Account
                </label>
                <Select
                  aria-label="Select Account"
                  placeholder="Choose Account"
                  value={localAccountId}
                  onChange={(key) => setLocalAccountId(key as string)}
                  className="w-full"
                >
                  <Select.Trigger className="bg-white/5 border-[0.5px] border-white/10 hover:border-primary/50 transition-all h-14 rounded-xl px-5 flex justify-between items-center w-full focus:ring-0 outline-none shadow-inner">
                    <Select.Value className="text-foreground text-[11px] font-black uppercase tracking-widest" />
                    <Select.Indicator className="text-foreground/30" />
                  </Select.Trigger>
                  <Select.Popover
                    className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl w-64 z-50 p-2"
                    placement="bottom left"
                  >
                    <ListBox>
                      {accounts.map((acc) => (
                        <ListBox.Item
                          key={acc.account_id}
                          id={acc.account_id}
                          textValue={acc.bank_name}
                          className="flex flex-col px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none transition-all"
                        >
                          <span className="text-foreground font-black text-[11px] uppercase tracking-tight italic">
                            {acc.bank_name}
                          </span>
                          <span className="text-foreground/20 text-[9px] font-mono tracking-widest mt-1.5 uppercase">
                            Account No: *{acc.account_number?.slice(-4)}
                          </span>
                        </ListBox.Item>
                      ))}
                    </ListBox>
                  </Select.Popover>
                </Select>
              </div>
              <Button
                onPress={() => setIsModalOpen(true)}
                className="w-full bg-white/5 hover:bg-white/10 text-foreground/60 hover:text-foreground border-[0.5px] border-white/10 rounded-xl h-14 font-black text-[10px] uppercase tracking-[0.2em] cursor-pointer transition-all shadow-lg"
              >
                Adjust Simulation
              </Button>
              <div className="pt-2">
                <p className="text-[8px] text-foreground/20 text-center leading-relaxed font-black uppercase tracking-[0.3em]">
                  Projections are based on historical data and AI models. Always verify with your actual bank balance.
                </p>
              </div>
            </div>
          </Card>
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
