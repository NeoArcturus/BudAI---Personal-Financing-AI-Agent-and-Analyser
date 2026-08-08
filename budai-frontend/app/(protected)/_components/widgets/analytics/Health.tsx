"use client";

import React, { useMemo, useState } from "react";
import { Card, Skeleton, CloseButton, ProgressCircle } from "@heroui/react";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard from "../../internal/FlipCard";
import { useHealthData } from "../../../_hooks/useHealthData";
import { useBudAI } from "@/app/context/AppContext";
import CoreChartEngine from "../../internal/ChartEngine";
import { buildChartConfig } from "../../../_utils/ChartBuilder";

export function Health() {
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts } = useBudAI();
  const localAccountId = accounts[0]?.account_id || null;
  
  const { healthData, isLoading, isError, isLocked } = useHealthData(localAccountId);

  const chartConfig = useMemo(() => {
    if (!healthData?.radarData) return null;
    return buildChartConfig(
      "health_radar",
      healthData.radarData,
      { bank_name_or_id: localAccountId || "ALL" },
      "Financial Health Radar",
    );
  }, [healthData?.radarData, localAccountId]);

  // Insight mapping for flip card
  const insights = useMemo(() => {
    if (!healthData?.metricsData?.recommendations) return undefined;
    return healthData.metricsData.recommendations.map(r => r.desc).join("\n\n");
  }, [healthData?.metricsData]);

  if (isLocked) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden border border-primary/20 bg-primary/5">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 animate-pulse">
                [ ANALYSIS SYNCING ]
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <div className="w-full h-full rounded-2xl bg-primary/10 animate-pulse border border-primary/20 flex items-center justify-center">
               <span className="text-[10px] font-mono text-primary/60 uppercase tracking-widest">Processing Latest Data...</span>
            </div>
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (isLoading) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Financial Health
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (isError || !healthData) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => {}}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Financial Health
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex flex-col items-center justify-center p-8 pt-0 relative overflow-hidden text-center">
            <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">Service Unavailable</p>
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  return (
    <WidgetFlipCard insight={insights} isLoading={false} onDiscuss={() => {}}>
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Financial Health
            </h3>
            <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
          </div>
        </div>

        <div className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-hidden">
          {healthData.metricsData && (
             <div className="absolute top-0 right-8 z-20 flex flex-col items-end">
                <span className="text-[9px] font-mono uppercase tracking-[0.3em] text-foreground/60 mb-2">
                  Overall Score
                </span>
                <ProgressCircle 
                  size="lg" 
                  classNames={{
                    base: "max-w-md",
                    track: "stroke-white/5",
                    indicator: "stroke-primary",
                  }}
                  value={isNaN(Number(healthData.metricsData.overall_score)) ? 0 : Number(healthData.metricsData.overall_score)} 
                />
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
        </div>
      </Card>
    </WidgetFlipCard>
  );
}
