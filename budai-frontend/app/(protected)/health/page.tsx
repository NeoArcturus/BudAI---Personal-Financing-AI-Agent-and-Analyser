"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { ArrowRightLeft, RefreshCcw, Zap } from "lucide-react";
import { Card, Skeleton } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";

export default function HealthPage() {
  const { accounts } = useBudAI();
  const [isLoading, setIsLoading] = useState(false);
  const [healthData, setHealthData] = useState(null);
  const [healthMetrics, setHealthMetrics] = useState<{
    overall_score: number;
    recommendations: { title: string; desc: string; type: string }[];
  } | null>(null);
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
      const [radarRes, metricsRes] = await Promise.all([
        apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "plot_health_radar",
              parameters: { bank_name_or_id: localAccountId },
            }),
          },
          true,
        ),
        apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "get_financial_health_metrics",
              parameters: { user_uuid: "CURRENT_USER" },
            }),
          },
          true,
        ),
      ]);

      if (radarRes.ok) {
        const result = await radarRes.json();
        setHealthData(result.data);
      }

      if (metricsRes.ok) {
        const result = await metricsRes.json();
        setHealthMetrics(
          typeof result.data === "string"
            ? JSON.parse(result.data)
            : result.data,
        );
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
    <div className="relative z-10 flex-1 flex flex-col pt-10 px-10 h-full">
      <div className="flex items-center justify-between mb-10 shrink-0">
        <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
          Financial <span className="font-normal not-italic">Health</span>
        </h2>
        <div className="flex items-center gap-4">
          <div className="bg-green-500/5 text-green-500 border-[0.5px] border-green-500/20 px-4 py-1.5 text-[9px] font-black uppercase tracking-[0.3em] shadow-sm rounded-lg">
            System Healthy
          </div>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8 pb-8 overflow-y-auto scrollbar-hide">
        <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
          <div className="flex items-center justify-between mb-10">
            <div className="flex items-center gap-5">
              <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-sm">
                <RefreshCcw size={20} />
              </div>
              <div>
                <h3 className="text-foreground font-black text-lg tracking-widest uppercase italic m-0">
                  Benchmarking
                </h3>
                <p className="text-foreground/30 text-[9px] font-black uppercase tracking-[0.4em] mt-1">
                  Industry Benchmarking
                </p>
              </div>
            </div>
          </div>

          <div className="flex-1 w-full flex items-center justify-center min-h-80 bg-white/1 rounded-xl border-[0.5px] border-white/5 relative overflow-hidden p-8">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.02)_0%,transparent_70%)] pointer-events-none" />
            {isLoading ? (
              <Skeleton animationType="shimmer" className="w-80 h-80 rounded-full bg-white/5" />
            ) : chartConfig ? (
              <CoreChartEngine config={chartConfig} />
            ) : (
              <div className="text-foreground/20 text-center opacity-40">
                <RefreshCcw
                  size={40}
                  className="mx-auto mb-4 animate-spin-slow"
                />
                <p className="text-[9px] font-black uppercase tracking-[0.4em]">
                  Syncing Financial Data...
                </p>
              </div>
            )}
          </div>
        </Card>

        <div className="flex flex-col gap-8">
          <Card className="liquid-glass rounded-xl p-10 shadow-inner flex flex-col items-center justify-center text-center border-none relative overflow-hidden">
            <div className="relative w-48 h-48 flex items-center justify-center mb-10">
              <div className="absolute inset-0 rounded-full border-[0.5px] border-white/5 shadow-inner"></div>
              {!isLoading && (
                <div className="absolute inset-0 rounded-full border-t-[3px] border-r-[3px] border-primary shadow-[0_0_30px_rgba(0,127,255,0.4)] rotate-45 animate-pulse"></div>
              )}
              <div className="flex flex-col items-center">
                {isLoading ? (
                  <Skeleton
                    animationType="shimmer"
                    className="w-16 h-12 rounded-lg bg-white/5"
                  />
                ) : (
                  <span className="text-6xl font-normal text-foreground tracking-tighter italic font-mono">
                    {healthMetrics?.overall_score || 0}
                  </span>
                )}
                <span className="text-[9px] text-foreground/30 font-black uppercase tracking-[0.4em] mt-3">
                  Financial Score
                </span>
              </div>
            </div>
            {isLoading ? (
              <div className="space-y-3 w-full max-w-xs px-4">
                <Skeleton
                  animationType="shimmer"
                  className="h-2 w-full rounded bg-white/5"
                />
                <Skeleton
                  animationType="shimmer"
                  className="h-2 w-3/4 rounded bg-white/5 mx-auto"
                />
              </div>
            ) : (
              <div className="bg-white/5 backdrop-blur-xl border-[0.5px] border-white/10 rounded-xl p-5 max-w-xs shadow-inner">
                <p className="text-foreground/60 text-[11px] leading-relaxed font-medium uppercase tracking-wide">
                  Your score is based on liquidity runway, debt drag, and net worth velocity.
                </p>
              </div>
            )}
          </Card>

          <Card className="liquid-glass rounded-xl p-10 shadow-inner flex-1 border-none">
            <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] mb-10 flex items-center gap-3 italic">
              <Zap size={18} className="text-primary shrink-0" />
              Recommendations
            </h4>
            <div className="space-y-5">
              {isLoading
                ? Array.from({ length: 3 }).map((_, i) => (
                    <div
                      key={i}
                      className="p-6 rounded-xl bg-white/5 border-[0.5px] border-white/10 space-y-3"
                    >
                      <Skeleton
                        animationType="shimmer"
                        className="h-3 w-1/2 rounded bg-white/5"
                      />
                      <Skeleton
                        animationType="shimmer"
                        className="h-2 w-full rounded bg-white/5"
                      />
                    </div>
                  ))
                : (healthMetrics?.recommendations || []).map((insight, i) => (
                    <div
                      key={i}
                      className="p-6 rounded-xl bg-white/2 border-[0.5px] border-white/5 hover:border-primary/40 transition-all cursor-pointer group shadow-sm"
                    >
                      <h5 className="text-foreground font-black text-[11px] mb-3 flex items-center justify-between tracking-widest uppercase">
                        {insight.title}
                        <ArrowRightLeft
                          size={14}
                          className="opacity-0 group-hover:opacity-100 transition-all -translate-x-2 group-hover:translate-x-0 text-primary"
                        />
                      </h5>
                      <p className="text-[10px] text-foreground/30 leading-relaxed font-bold uppercase tracking-tight">
                        {insight.desc}
                      </p>
                    </div>
                  ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
