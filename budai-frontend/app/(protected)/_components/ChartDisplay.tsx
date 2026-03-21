// app/dashboard/_components/ChartDisplay.tsx
"use client";

import React from "react";
import { Loader2, BarChart3 } from "lucide-react";
import DynamicChart from "./DynamicChart";
import { NativeChartConfig } from "@/types";

interface ChartDisplayProps {
  isGenerating: boolean;
  chartConfig: NativeChartConfig | null;
}

export const ChartDisplay: React.FC<ChartDisplayProps> = ({
  isGenerating,
  chartConfig,
}) => {
  return (
    <div className="flex-1 relative p-4 flex flex-col items-center justify-center">
      {isGenerating ? (
        <div className="flex flex-col items-center text-slate-500 space-y-4">
          <Loader2 className="w-12 h-12 animate-spin text-[#00FFAA]" />
          <p className="text-xs tracking-widest uppercase font-bold text-[#00FFAA] animate-pulse">
            Generating Analysis...
          </p>
        </div>
      ) : chartConfig ? (
        <DynamicChart config={chartConfig} />
      ) : (
        <div className="text-center max-w-md">
          <BarChart3 className="w-16 h-16 mb-4 opacity-50 mx-auto text-[#00FFAA]" />
          <p className="text-sm tracking-widest uppercase font-bold text-slate-400">
            Select a Dashboard
          </p>
          <p className="text-xs opacity-50 mt-2">
            Click one of the quick action buttons above to view your charts
            instantly, or ask BudAI for a custom analysis below.
          </p>
        </div>
      )}
    </div>
  );
};
