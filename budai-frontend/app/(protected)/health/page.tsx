"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  LayoutDashboard,
  ArrowRightLeft,
  RefreshCcw,
  CreditCard,
  Zap,
  LineChart,
} from "lucide-react";
import { Link, Avatar, Card, Spinner, Chip } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";

export default function HealthPage() {
  const { userName, accounts } = useBudAI();
  const [isLoading, setIsLoading] = useState(false);
  const [healthData, setHealthData] = useState(null);
  const [localAccountId, setLocalAccountId] = useState<string>("");

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  const fetchHealth = useCallback(async () => {
    if (!localAccountId) return;
    setIsLoading(true);
    try {
      const response = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          body: JSON.stringify({
            tool_name: "plot_health_radar",
            parameters: {
              bank_name_or_id: localAccountId,
            },
          }),
        },
        true,
      );
      if (response.ok) {
        const result = await response.json();
        setHealthData(result.data);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  }, [localAccountId]);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  const chartConfig = useMemo(() => {
    if (!healthData) return null;
    return buildChartConfig(
      "health_radar",
      healthData,
      { bank_name_or_id: localAccountId },
      "Financial Health Radar",
    );
  }, [healthData, localAccountId]);

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
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <LineChart size={20} />
              <span className="font-medium text-sm">Forecasting</span>
            </Link>
            <Link
              href="/health"
              className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20"
            >
              <RefreshCcw size={20} />
              <span className="font-semibold text-sm">Health Radar</span>
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
          <h2 className="text-white text-3xl font-bold tracking-tight">
            Health Radar
          </h2>
          <div className="flex items-center gap-4">
            <Chip className="bg-brand-green/10 text-brand-green border border-brand-green/20 px-3 py-1 text-xs font-bold uppercase tracking-widest">
              Status: Excellent
            </Chip>
          </div>
        </div>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 pb-8 overflow-y-auto [&::-webkit-scrollbar]:hidden">
          <Card className="obsidian-glass rounded-3xl p-6 min-h-125 flex flex-col shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-neon-cyan/10 flex items-center justify-center text-neon-cyan">
                  <RefreshCcw size={20} />
                </div>
                <div>
                  <h3 className="text-white font-bold text-lg">
                    Financial Benchmarking
                  </h3>
                  <p className="text-[#8B8E98] text-xs">
                    Comparison against peer averages across 5 dimensions
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
                  <RefreshCcw size={48} className="mx-auto mb-4 opacity-20" />
                  <p>Analyzing financial health metrics...</p>
                </div>
              )}
            </div>
          </Card>

          <div className="flex flex-col gap-6">
            <Card className="obsidian-glass rounded-3xl p-8 shadow-2xl flex flex-col items-center justify-center text-center">
              <div className="relative w-40 h-40 flex items-center justify-center mb-6">
                <div className="absolute inset-0 rounded-full border-4 border-white/5"></div>
                <div className="absolute inset-0 rounded-full border-t-4 border-r-4 border-neon-cyan shadow-[0_0_20px_rgba(0,229,255,0.4)] rotate-45"></div>
                <div className="flex flex-col items-center">
                  <span className="text-5xl font-black text-white">84</span>
                  <span className="text-xs text-[#8B8E98] font-bold uppercase tracking-tighter mt-1">
                    Health Score
                  </span>
                </div>
              </div>
              <p className="text-[#8B8E98] text-sm max-w-xs leading-relaxed">
                Your score increased by{" "}
                <span className="text-brand-green font-bold">+4 points</span>{" "}
                this month due to improved debt-to-income ratio.
              </p>
            </Card>

            <Card className="obsidian-glass rounded-3xl p-6 shadow-2xl flex-1">
              <h4 className="text-white font-bold mb-6 flex items-center gap-2">
                <Zap size={18} className="text-neon-cyan" />
                Actionable Insights
              </h4>
              <div className="space-y-4">
                {[
                  {
                    title: "Increase Emergency Fund",
                    desc: "You have 1.5 months of expenses saved. Goal is 3 months.",
                    type: "urgent",
                  },
                  {
                    title: "Consolidate Debt",
                    desc: "High interest on your credit card is dragging down your score.",
                    type: "warning",
                  },
                  {
                    title: "Investment Diversity",
                    desc: "90% of your assets are in cash. Consider low-risk index funds.",
                    type: "info",
                  },
                ].map((insight, i) => (
                  <div
                    key={i}
                    className="p-4 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 transition-all cursor-pointer group"
                  >
                    <h5 className="text-white font-bold text-sm mb-1 flex items-center justify-between">
                      {insight.title}
                      <ArrowRightLeft
                        size={14}
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-neon-cyan"
                      />
                    </h5>
                    <p className="text-xs text-[#8B8E98] leading-relaxed">
                      {insight.desc}
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
