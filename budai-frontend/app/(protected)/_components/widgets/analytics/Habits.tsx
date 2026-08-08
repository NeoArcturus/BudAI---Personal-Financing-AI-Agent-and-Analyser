"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, Skeleton, CloseButton } from "@heroui/react";
import { apiFetch, getUserUuid } from "@/lib/api";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard from "../../internal/FlipCard";
import CoreChartEngine from "../../internal/ChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";

interface HabitCluster {
  name: string;
  total_spend: number;
  transaction_count: number;
}

interface HabitsData {
  clusters: HabitCluster[];
}

export function Habits() {
  const { onRemove } = React.useContext(WidgetContext);
  const [data, setData] = useState<HabitsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const uuid = getUserUuid();
        if (!uuid) throw new Error("No user uuid");
        const res = await apiFetch(`/api/analytics/clusters?user_uuid=${uuid}`, {}, true);
        if (!res.ok) throw new Error("Failed to fetch");
        const json = await res.json() as HabitsData;
        setData(json);
      } catch (err) {
        console.error("Habits widget error:", err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const chartConfig = useMemo(() => {
    if (!data?.clusters) return null;
    const payload = [
      {
        bank_name: "All Accounts",
        data: data.clusters.map((c) => ({
          Category: c.name,
          Total_Amount: c.total_spend,
        })),
      },
    ];
    return buildChartConfig(
      "categorized",
      payload,
      {},
      "Spending Habits"
    );
  }, [data]);

  const headerContent = (
    <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
      <div className="flex justify-between items-start w-full">
        <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
          Spending Habits
        </h3>
        <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
      </div>
    </Card.Header>
  );

  if (loading) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          {headerContent}
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
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          {headerContent}
          <Card.Content className="flex-1 w-full flex flex-col items-center justify-center p-8 pt-0 relative overflow-hidden text-center">
            <p className="text-foreground/40 text-[10px] font-mono uppercase tracking-[0.2em]">Analytics Unavailable</p>
          </Card.Content>
        </Card>
      </WidgetFlipCard>
    );
  }

  return (
    <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => {}}>
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        {headerContent}

        <Card.Content className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-hidden">
          <div className="flex-1 w-full h-full min-h-[250px]">
            <CoreChartEngine config={chartConfig} />
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
