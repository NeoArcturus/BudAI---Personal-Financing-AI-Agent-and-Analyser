"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, Skeleton, CloseButton } from "@heroui/react";
import { apiFetch, getUserUuid } from "@/lib/api";
import CoreChartEngine from "../../internal/ChartEngine";
import { ChartConfiguration } from "chart.js";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard from "../../internal/FlipCard";

interface RiskData {
  most_vulnerable_day: string;
  historical_density_spend: number;
}

export function Risk() {
  const { onRemove } = React.useContext(WidgetContext);
  const [data, setData] = useState<RiskData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const uuid = getUserUuid();
        if (!uuid) throw new Error("No user uuid");
        const res = await apiFetch(`/api/analytics/impulse-vulnerability?user_uuid=${uuid}`, {}, true);
        if (!res.ok) throw new Error("Failed to fetch");
        const json = await res.json() as RiskData;
        setData(json);
      } catch (err) {
        console.error("Risk widget error:", err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency: "GBP",
      maximumFractionDigits: 0,
    }).format(val);

  const chartConfig = useMemo<ChartConfiguration | null>(() => {
    if (!data) return null;

    return {
      type: "bar",
      data: {
        labels: [data.most_vulnerable_day.toUpperCase()],
        datasets: [
          {
            label: "Historical Density",
            data: [data.historical_density_spend],
            backgroundColor: "hsl(var(--warning))",
            borderRadius: 6,
            borderWidth: 0,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(0,0,0,0.8)",
            titleColor: "rgba(255,255,255,0.7)",
            bodyColor: "#fff",
            bodyFont: { family: "monospace", size: 12 },
            borderColor: "hsla(var(--warning), 0.2)",
            borderWidth: 1,
            displayColors: false,
            callbacks: {
              label: (context: any) => {
                const val = context.parsed.x;
                return new Intl.NumberFormat("en-GB", {
                  style: "currency",
                  currency: "GBP",
                }).format(val);
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: {
              color: "rgba(255,255,255,0.4)",
              font: { family: "monospace", size: 9 },
              callback: (value: any) => "£" + value,
            },
          },
          y: {
            grid: { display: false },
            ticks: {
              color: "hsla(var(--foreground), 0.8)",
              font: { family: "monospace", size: 10, weight: "bold" },
            },
          },
        },
      },
    };
  }, [data]);

  if (loading) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Spend Volatility
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </Card.Header>
          <Card.Content className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
          </Card.Content>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (error || !chartConfig) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Spend Volatility
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </Card.Header>
          <Card.Content className="flex-1 w-full flex flex-col items-center justify-center p-8 pt-0 relative overflow-hidden text-center">
            <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">Service Unavailable</p>
          </Card.Content>
        </Card>
      </WidgetFlipCard>
    );
  }

  return (
    <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => {}}>
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Spend Volatility
            </h3>
            <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
          </div>
        </Card.Header>

        <Card.Content className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-hidden">
          <div className="w-full h-full relative z-10 min-h-[250px]">
             <CoreChartEngine config={chartConfig} />
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
