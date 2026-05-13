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
} from "lucide-react";
import { Button, Link, Avatar, Card, Spinner } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";

export default function ForecastingPage() {
  const { userName, accounts } = useBudAI();
  const [isLoading, setIsLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [localAccountId, setLocalAccountId] = useState<string>("");

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  const fetchForecast = useCallback(async () => {
    if (!localAccountId) return;
    setIsLoading(true);
    try {
      const response = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          body: JSON.stringify({
            tool_name: "generate_financial_forecast",
            parameters: {
              bank_name_or_id: localAccountId,
              horizon_months: 12,
            },
          }),
        },
        true,
      );
      if (response.ok) {
        const result = await response.json();
        setForecastData(result.data);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  }, [localAccountId]);

  useEffect(() => {
    fetchForecast();
  }, [fetchForecast]);

  const chartConfig = useMemo(() => {
    if (!forecastData) return null;
    return buildChartConfig(
      "balance_forecast",
      forecastData,
      { bank_name_or_id: localAccountId },
      "Wealth Trajectory",
    );
  }, [forecastData, localAccountId]);

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">
      {/* Background Glows */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/5 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/5 blur-[180px]"></div>
      </div>

      {/* Sidebar */}
      <div className="relative z-10 w-64 h-full bg-obsidian/40 backdrop-blur-xl border-r border-white/8 flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
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

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <h2 className="text-white text-3xl font-bold tracking-tight">
            Wealth Forecasting
          </h2>
          <div className="flex items-center gap-4">
            <Button
              onPress={fetchForecast}
              className="bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30 hover:bg-neon-cyan/20 rounded-xl flex items-center gap-2 font-bold px-6"
            >
              <Sparkles size={18} />
              Recalculate Projections
            </Button>
          </div>
        </div>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 pb-8 overflow-y-auto [&::-webkit-scrollbar]:hidden">
          <Card className="lg:col-span-2 bg-obsidian/40 backdrop-blur-xl border border-white/8 rounded-3xl p-6 min-h-125 flex flex-col shadow-2xl">
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
                    AI-projected wealth accumulation for the next 12 months
                  </p>
                </div>
              </div>
            </div>

            <div className="flex-1 w-full flex items-center justify-center">
              {isLoading ? (
                <Spinner color="accent" />
              ) : chartConfig ? (
                <CoreChartEngine config={chartConfig} />
              ) : (
                <div className="text-[#5E6272] text-center">
                  <Clock size={48} className="mx-auto mb-4 opacity-20" />
                  <p>Select an account to generate forecast</p>
                </div>
              )}
            </div>
          </Card>

          <div className="space-y-6">
            <Card className="bg-obsidian/40 backdrop-blur-xl border border-white/8 rounded-3xl p-6 shadow-2xl">
              <h4 className="text-white font-bold mb-4">AI Snapshot</h4>
              <div className="space-y-4">
                <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                  <p className="text-xs text-[#8B8E98] uppercase tracking-wider font-bold mb-1">
                    Projected Net Worth
                  </p>
                  <p className="text-2xl font-bold text-neon-cyan">
                    £156,240.00
                  </p>
                  <p className="text-xs text-brand-green mt-1 flex items-center gap-1">
                    <TrendingUp size={12} />
                    +12.4% from current
                  </p>
                </div>
                <div className="p-4 rounded-2xl bg-neon-cyan/5 border border-neon-cyan/20">
                  <p className="text-sm text-white font-medium leading-relaxed">
                    Based on your current spending trends, you&apos;re on track
                    to hit your savings goal 2 months early.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="bg-obsidian/40 backdrop-blur-xl border border-white/8 rounded-3xl p-6 shadow-2xl">
              <h4 className="text-white font-bold mb-4">Forecast Settings</h4>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs text-[#8B8E98] font-bold uppercase tracking-wider">
                    Target Account
                  </label>
                  <div className="bg-[#181A20] border border-white/8 rounded-xl px-4 py-3 text-white text-sm flex justify-between items-center">
                    <span>
                      {accounts.find((a) => a.account_id === localAccountId)
                        ?.bank_name || "Select Account"}
                    </span>
                    <ArrowRightLeft size={16} className="text-[#5E6272]" />
                  </div>
                </div>
                <Button className="w-full bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl h-12 font-medium">
                  Adjust Savings Rate
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
