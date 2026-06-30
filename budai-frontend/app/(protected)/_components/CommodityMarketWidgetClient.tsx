
"use client";

import React, { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Globe, TrendingUp, TrendingDown, ChevronDown } from "lucide-react";
import {
  Card,
  Skeleton,
  ToggleButton,
  ToggleButtonGroup,
  Text,
  Dropdown,
  Selection,
} from "@heroui/react";
import { cn } from "@/lib/utils";
import CoreChartEngine from "./CoreChartEngine";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { useBudAI } from "@/app/context/AppContext";

interface CommodityHistory {
  symbol: string;
  history: { Date: string; Close: number }[];
}

interface CommodityMarketWidgetProps {
  initialHistory?: { history: CommodityHistory[] };
}

const RANGES = ["1D", "1W", "1M", "3M", "6M", "1Y"];

export default function CommodityMarketWidgetClient({
  initialHistory,
}: CommodityMarketWidgetProps) {
  const router = useRouter();
  const { createNewSession } = useBudAI();
  const [selectedSymbol, setSelectedSymbol] = useState<string>("GC=F");
  const [selectedRange, setSelectedRange] = useState<string>("1M");

  const { data: historyData, isLoading: isHistoryLoading } = useQuery<{
    history: CommodityHistory[];
  }>({
    queryKey: ["market-history", selectedRange],
    queryFn: async () => {
      const res = await apiFetch(
        `/api/market/history?range=${selectedRange}`,
        {},
        true,
      );
      return (await res.json()) as { history: CommodityHistory[] };
    },
    initialData: selectedRange === "1M" ? initialHistory : undefined,
    staleTime: 600000,
  });

  const symbols = useMemo(() => {
    if (!historyData?.history) return [];
    return historyData.history.map((h) => h.symbol);
  }, [historyData]);

  const activeHistory = useMemo(() => {
    return historyData?.history.find((h) => h.symbol === selectedSymbol);
  }, [historyData, selectedSymbol]);

  const isPositive = useMemo(() => {
    if (!activeHistory || activeHistory.history.length < 2) return true;
    const firstPrice = activeHistory.history[0].Close;
    const lastPrice =
      activeHistory.history[activeHistory.history.length - 1].Close;
    return lastPrice >= firstPrice;
  }, [activeHistory]);

  const trendColor = isPositive ? "#22c55e" : "#ef4444";
  const trendBg = isPositive
    ? "rgba(34, 197, 94, 0.1)"
    : "rgba(239, 68, 68, 0.1)";

  const chartConfig = useMemo(() => {
    if (!activeHistory) return null;

    const labels = activeHistory.history.map((pt) => pt.Date);
    const data = activeHistory.history.map((pt) => pt.Close);

    return {
      type: "line" as const,
      data: {
        labels,
        datasets: [
          {
            label: `${selectedSymbol.replace("=F", "").replace("=X", "")} Price`,
            data: data,
            borderColor: trendColor,
            backgroundColor: trendBg,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: { color: "rgba(255, 255, 255, 0.8)", font: { size: 10 } },
          },
        },
      },
    };
  }, [activeHistory, selectedSymbol, trendColor, trendBg]);

  const handleDiscuss = () => {
    const sessionId = createNewSession(
      `Market Analysis: ${selectedSymbol} (${selectedRange})`,
      {
        type: "market_audit",
        data: {
          symbol: selectedSymbol,
          range: selectedRange,
          history: JSON.stringify(activeHistory || {}),
        },
      },
    );
    router.push(`/advisor?session=${sessionId}`);
  };

  const handleRangeChange = (keys: Selection) => {
    const first = Array.from(keys)[0];
    if (first) setSelectedRange(first as string);
  };

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      isDataLoading={isHistoryLoading}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full liquid-glass border-none rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="p-8 border-b-[0.5px] border-white/5 shrink-0 flex items-center justify-between z-10">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,242,255,0.05)]">
              <Globe size={20} />
            </div>
            <div className="flex flex-col">
              <h3 className="text-[10px] font-black text-foreground uppercase tracking-[0.4em] italic m-0">
                Global Commodity Market
              </h3>
              <p className="text-primary/50 text-[8px] font-black uppercase tracking-[0.3em] mt-1.5">
                Market Data Insights
              </p>
            </div>
          </div>
        </Card.Header>

        <Card.Content className="p-0 flex-1 flex flex-col overflow-hidden min-h-0 relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.01)_0%,transparent_70%)] pointer-events-none" />
          <div className="flex-1 flex flex-col p-8 gap-8 overflow-y-auto scrollbar-hide relative z-10">
            <div className="flex flex-col gap-3 shrink-0">
              <span className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                Target Asset
              </span>
              <ToggleButtonGroup
                disallowEmptySelection
                className="bg-white/5 backdrop-blur-md p-1 rounded-xl border-[0.5px] border-white/10 w-full flex flex-row gap-1 shadow-inner"
                selectedKeys={new Set([selectedSymbol])}
                selectionMode="single"
                size="sm"
                onSelectionChange={(keys) => {
                  const first = Array.from(keys)[0];
                  if (first) setSelectedSymbol(first as string);
                }}
              >
                {symbols.map((s) => (
                  <ToggleButton
                    key={s}
                    id={s}
                    variant="ghost"
                    className={cn(
                      "flex-1 py-2 rounded-lg text-[9px] font-black uppercase tracking-tighter transition-all h-auto w-10 border-none",
                      selectedSymbol === s
                        ? "bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white border-none shadow-lg"
                        : "text-foreground/40 hover:text-foreground data-[hovered=true]:bg-white/10",
                    )}
                  >
                    {s.replace("=X", "").replace("=F", "").replace("^", "")}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </div>

            <div className="flex-1 flex flex-col gap-4 min-h-0">
              <div className="flex items-center justify-between px-1 shrink-0">
                <div className="flex items-center gap-3">
                  {isPositive ? (
                    <TrendingUp size={14} className="text-green-500" />
                  ) : (
                    <TrendingDown size={14} className="text-primary" />
                  )}
                  <span className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/60 italic">
                    Data Stream
                  </span>
                </div>
                {activeHistory && (
                  <div
                    className="px-3 py-1 rounded-lg border-[0.5px]"
                    style={{
                      backgroundColor: `${trendColor}11`,
                      borderColor: `${trendColor}33`,
                    }}
                  >
                    <span
                      className="font-mono text-[11px] font-black tracking-tighter"
                      style={{ color: trendColor }}
                    >
                      £
                      {activeHistory.history[
                        activeHistory.history.length - 1
                      ]?.Close.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex-1 relative min-h-40 bg-white/1 rounded-xl border-[0.5px] border-white/10 p-6 overflow-hidden shadow-inner flex items-center justify-center">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.02)_0%,transparent_70%)] pointer-events-none" />
                {isHistoryLoading ? (
                  <Skeleton
                    animationType="shimmer"
                    className="w-full h-full rounded-lg bg-white/5"
                  />
                ) : chartConfig ? (
                  <CoreChartEngine config={chartConfig} />
                ) : (
                  <div className="flex flex-col items-center gap-3 opacity-20">
                    <Globe size={24} className="text-foreground" />
                    <span className="text-[9px] font-black uppercase tracking-[0.3em]">
                      Awaiting Data Sync
                    </span>
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-3 shrink-0">
                <span className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30 pl-1 italic">
                  Time Snapshots
                </span>
                <Dropdown>
                  <Dropdown.Trigger className="w-full justify-between bg-white/5 border-[0.5px] border-white/10 hover:border-primary/50 transition-all rounded-xl px-5 py-3 text-[10px] font-black uppercase tracking-[0.2em] flex flex-row cursor-pointer outline-none shadow-inner">
                    {selectedRange}
                    <ChevronDown size={14} className="text-foreground/30" />
                  </Dropdown.Trigger>
                  <Dropdown.Popover className="min-w-40 bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 shadow-2xl rounded-xl z-50">
                    <Dropdown.Menu
                      selectionMode="single"
                      selectedKeys={new Set([selectedRange])}
                      onSelectionChange={handleRangeChange}
                      className="p-2"
                    >
                      {RANGES.map((r) => (
                        <Dropdown.Item
                          key={r}
                          id={r}
                          textValue={r}
                          className={cn(
                            "rounded-lg px-4 py-2.5 transition-all outline-none border-[0.5px] border-transparent",
                            selectedRange === r
                              ? "bg-primary/20 text-primary border-primary/30"
                              : "text-foreground/40 hover:bg-white/10 hover:text-foreground",
                          )}
                        >
                          <span className="text-[9px] font-black uppercase tracking-widest font-mono">
                            {r}
                          </span>
                        </Dropdown.Item>
                      ))}
                    </Dropdown.Menu>
                  </Dropdown.Popover>
                </Dropdown>
              </div>
            </div>
          </div>
          <div className="p-4 border-t-[0.5px] border-white/5 bg-white/1 relative z-10">
            <Text className="text-center text-[8px] text-foreground/20 font-black uppercase tracking-[0.5em]">
              Real-time Global Market sync
            </Text>
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
