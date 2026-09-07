"use client";

import React, { useMemo, useState, useEffect } from "react";
import { Card, Skeleton, ProgressCircle, Meter, Label } from "@heroui/react";
import { Activity, Target, Repeat, CreditCard, Heart, PieChart } from "lucide-react";
import CoreChartEngine from "@/app/(protected)/_components/internal/ChartEngine";
import { useHealthData } from "@/app/(protected)/_hooks/useHealthData";
import { useExpenseCategories } from "@/lib/hooks";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { apiFetch, getUserUuid } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { AnimatedNumber } from "@/app/(protected)/_components/ui/AnimatedNumber";
import { today, getLocalTimeZone } from "@internationalized/date";

// Custom Card Wrapper for the Dashboard aesthetic without flip functionality
function ConnectionCard({ title, icon: Icon, children }: any) {
  return (
    <Card className="w-full h-full bg-content1 border border-white/5 shadow-xl">
      <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
        <div className="flex justify-between items-start w-full">
          <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 flex items-center gap-2">
            <Icon size={12} /> {title}
          </h3>
        </div>
      </Card.Header>
      <Card.Content className="flex-1 w-full min-h-0 p-8 pt-0 relative overflow-y-auto custom-scrollbar">
        {children}
      </Card.Content>
    </Card>
  );
}

export function ConnectionHealthWidget({ accountId }: { accountId: string }) {
  const { healthData, isLoading, isError, isLocked } = useHealthData(accountId);

  const chartConfig = useMemo(() => {
    if (!healthData?.radarData) return null;
    return buildChartConfig("health_radar", healthData.radarData, { bank_name_or_id: accountId }, "Financial Health");
  }, [healthData?.radarData, accountId]);

  if (isLocked) {
    return (
      <ConnectionCard title="Financial Health" icon={Heart}>
        <div className="w-full h-full rounded-2xl bg-primary/10 animate-pulse border border-primary/20 flex items-center justify-center">
          <span className="text-[10px] font-mono text-primary/60 uppercase tracking-widest">Processing Latest Data...</span>
        </div>
      </ConnectionCard>
    );
  }

  if (isLoading || isError || !healthData) {
    return (
      <ConnectionCard title="Financial Health" icon={Heart}>
        <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
      </ConnectionCard>
    );
  }

  return (
    <ConnectionCard title="Financial Health" icon={Heart}>
      {healthData.metricsData && (
        <div className="absolute top-0 right-8 z-20 flex flex-col items-end">
          <span className="text-[9px] font-mono uppercase tracking-[0.3em] text-foreground/60 mb-2">
            Overall Score
          </span>
          <ProgressCircle
            size="lg"
            className="max-w-md"
            value={isNaN(Number(healthData.metricsData.overall_score)) ? 0 : Number(healthData.metricsData.overall_score)}
          >
            <ProgressCircle.Track>
              <ProgressCircle.TrackCircle className="stroke-white/5" />
              <ProgressCircle.FillCircle className="stroke-primary" />
            </ProgressCircle.Track>
          </ProgressCircle>
          <span className="text-xl font-black font-mono text-primary mt-1">
            {isNaN(Number(healthData.metricsData.overall_score)) ? 0 : Number(healthData.metricsData.overall_score)}
          </span>
        </div>
      )}
      {chartConfig ? (
        <div className="w-full h-full mt-4">
          <CoreChartEngine config={chartConfig} />
        </div>
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">No Data Available</span>
        </div>
      )}
    </ConnectionCard>
  );
}

export function ConnectionExpenseWidget({ accountId }: { accountId: string }) {
  const endDate = today(getLocalTimeZone());
  const startDate = endDate.subtract({ months: 1 });
  const fromStr = `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`;
  const toStr = `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`;

  const { data, isFetching } = useExpenseCategories(accountId, fromStr, toStr, true);

  const config = useMemo(() => {
    if (!data || data.length === 0 || !data[0].data) return null;
    const aggregated = data[0].data
      .map((item: any) => ({
        name: item.Category || item.category || item.Date || item.date || "Other",
        value: Number(item.Total_Amount || item.total_amount || item.Amount || item.amount || 0),
      }))
      .filter((i: any) => i.value > 0)
      .sort((a: any, b: any) => b.value - a.value);

    if (aggregated.length === 0) return null;

    return buildChartConfig("categorized_doughnut", [{
      bank_name: "Expenses",
      data: aggregated.map((d: any) => ({ Category: d.name, Total_Amount: d.value }))
    }], {}, "Distribution");
  }, [data]);

  return (
    <ConnectionCard title="Expense Distribution" icon={PieChart}>
      {isFetching ? (
        <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
      ) : config ? (
        <div className="w-full h-full relative">
          <CoreChartEngine config={config} />
        </div>
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">No Data Available</span>
        </div>
      )}
    </ConnectionCard>
  );
}

export function ConnectionHabitsWidget({ accountId }: { accountId: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const uuid = getUserUuid();
        const res = await apiFetch(`/api/analytics/clusters?user_uuid=${uuid}&account_id=${accountId}`, {}, true);
        const json = await res.json();
        setData(json);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    if (accountId) fetchData();
  }, [accountId]);

  const config = useMemo(() => {
    if (!data?.clusters || data.clusters.length === 0) return null;
    return buildChartConfig("categorized", [{
      bank_name: "Habits",
      data: data.clusters.map((c: any) => ({
        Category: c.name || c.merchant_name || "Unknown",
        Total_Amount: c.total_spend || c.total_amount || 0,
      }))
    }], {}, "Spending Habits");
  }, [data]);

  return (
    <ConnectionCard title="Spending Habits" icon={Activity}>
      {loading ? (
        <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
      ) : config ? (
        <div className="w-full h-full">
          <CoreChartEngine config={config} />
        </div>
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">No Data Available</span>
        </div>
      )}
    </ConnectionCard>
  );
}

export function ConnectionBucketWidget({ accountId, type, title, icon }: { accountId: string, type: string, title: string, icon: any }) {
  const { data: buckets = [], isLoading } = useQuery({
    queryKey: ["buckets"],
    queryFn: () => apiFetch("/api/buckets").then(res => res.json())
  });

  const filteredBuckets = useMemo(() => {
    const arr = Array.isArray(buckets) ? buckets : (buckets as any)?.buckets || (buckets as any)?.data || [];
    return arr.filter((b: any) => b.type === type || b.bucket_type === type);
  }, [buckets, type]);

  return (
    <ConnectionCard title={title} icon={icon}>
      {isLoading ? (
        <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
      ) : filteredBuckets.length === 0 ? (
        <div className="w-full h-full flex items-center justify-center">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">No Buckets Found</span>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {filteredBuckets.map((b: any, idx: number) => (
            <div key={b.id || b.bucket_id || idx} className="flex flex-col p-4 bg-white/5 rounded-xl border border-white/5">
              <div className="flex justify-between items-center">
              <div>
                <p className="font-bold text-sm tracking-wide text-foreground truncate">{b.name}</p>
                <p className="text-[9px] font-mono text-foreground/50 uppercase tracking-widest">{b.status || "ACTIVE"}</p>
              </div>
              <div className="text-right flex flex-col items-end">
                <p className="font-mono text-sm tracking-tighter text-foreground">
                  <AnimatedNumber value={b.cached_balance || 0} minimumFractionDigits={2} maximumFractionDigits={2} currency={b.currency} />
                </p>
                {b.target_amount && (
                  <p className="text-[9px] font-mono text-foreground/40">
                    / £{b.target_amount}
                  </p>
                )}
              </div>
            </div>
            {b.target_amount > 0 && (
              <Meter
                aria-label={`${b.name} progress`}
                value={Math.min(100, Math.max(0, ((b.cached_balance || 0) / b.target_amount) * 100))}
                className="w-full mt-2"
              >
                <div className="flex justify-between text-[9px] uppercase tracking-widest font-black text-foreground/40 mb-1 w-full">
                  <Label>Progress</Label>
                  <Meter.Output />
                </div>
                <Meter.Track className="bg-white/10 h-1.5 w-full rounded-full overflow-hidden">
                  <Meter.Fill className="bg-primary rounded-full" />
                </Meter.Track>
              </Meter>
            )}
            </div>
          ))}
        </div>
      )}
    </ConnectionCard>
  );
}
