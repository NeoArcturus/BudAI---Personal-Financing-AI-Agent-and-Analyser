"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, Skeleton, CloseButton } from "@heroui/react";
import { apiFetch, getUserUuid } from "@/lib/api";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard from "../../internal/FlipCard";
import CoreChartEngine from "../../internal/ChartEngine";
import { ChartConfiguration } from "chart.js";

interface AnomalyItem {
  transaction_uuid: string;
  merchant: string;
  amount: number;
  date: string;
  reason: string;
}

interface AnomaliesData {
  anomalies: AnomalyItem[];
}

export function Anomalies() {
  const { onRemove } = React.useContext(WidgetContext);
  const [data, setData] = useState<AnomaliesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const uuid = getUserUuid();
        if (!uuid) throw new Error("No user uuid");
        const res = await apiFetch(`/api/analytics/anomalies?user_uuid=${uuid}`, {}, true);
        if (!res.ok) throw new Error("Failed to fetch");
        const json = await res.json() as AnomaliesData;
        setData(json);
      } catch (err) {
        console.error("Anomalies widget error:", err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const chartConfig = useMemo(() => {
    if (!data?.anomalies) return null;

    const scatterData = data.anomalies.map((anomaly, idx) => ({
      x: idx + 1, // simplified time mapping for visual
      y: anomaly.amount,
      merchant: anomaly.merchant,
      date: anomaly.date,
    }));

    return {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Irregularities",
            data: scatterData,
            backgroundColor: "hsl(var(--danger))",
            borderColor: "hsl(var(--danger))",
            pointRadius: 6,
            pointHoverRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context: any) => {
                const pt = context.raw;
                const curr = new Intl.NumberFormat("en-GB", { style: "currency", currency: pt.currency || "GBP" }).formatToParts(1).find(x => x.type === "currency")?.value || "£";
                return `${pt.merchant}: ${curr}${pt.y.toFixed(2)}`;
              },
            },
          },
        },
        scales: {
          x: {
            display: false, // hide arbitrary x axis
          },
          y: {
            grid: { color: "rgba(255,255,255,0.05)" },
            border: { display: false },
            ticks: {
              color: "rgba(255,255,255,0.4)",
              font: { family: "monospace", size: 10 },
            }
          },
        },
      },
    } as ChartConfiguration;
  }, [data]);

  if (loading) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => {}}>
        <Card className="w-full h-full">
          <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Irregular Activity
              </h3>
              <CloseButton onPress={onRemove} className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full" />
            </div>
          </Card.Header>
          <Card.Content className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
          </Card.Content>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (error || !data || !chartConfig) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => {}}>
        <Card className="w-full h-full">
          <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Irregular Activity
              </h3>
              <CloseButton onPress={onRemove} className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full" />
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
      <Card className="w-full h-full">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Irregular Activity
            </h3>
            <CloseButton onPress={onRemove} className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full" />
          </div>
        </Card.Header>

        <Card.Content className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-hidden">
          {data.anomalies.length === 0 ? (
              <div className="flex-1 border border-white/5 bg-white/[0.02] rounded-xl flex items-center justify-center text-center">
                <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">
                  No Irregularities Detected
                </span>
              </div>
          ) : (
            <div className="flex-1 w-full h-full min-h-[250px]">
              <CoreChartEngine config={chartConfig} />
            </div>
          )}
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
