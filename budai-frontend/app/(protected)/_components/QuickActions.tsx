"use client";

import React from "react";
import { TabType } from "@/types";

interface QuickActionPanelProps {
  isGenerating: boolean;
  activeAccountId: string | null;
  onTriggerChart: (chartId: TabType | string) => void;
}

export const QuickActionPanel: React.FC<QuickActionPanelProps> = ({
  isGenerating,
  activeAccountId,
  onTriggerChart,
}) => {
  // Hide buttons if "ALL" accounts view is active
  if (activeAccountId === "ALL" || !activeAccountId) {
    return null;
  }

  const quickActionCharts = [
    "historical_monthly",
    "categorized",
    "cash_flow_mixed",
    "health_radar",
    "expense_forecast",
    "balance_forecast",
  ];

  const getChartDisplayName = (chartId: string) => {
    switch (chartId) {
      case "historical_daily":
        return "Daily Spend";
      case "historical_weekly":
        return "Weekly Spend";
      case "historical_monthly":
        return "Monthly History";
      case "categorized":
        return "Category Breakdown";
      case "cash_flow_mixed":
        return "Income vs Expenses";
      case "health_radar":
        return "Health Radar";
      case "expense_forecast":
        return "Expense Forecast";
      case "balance_forecast":
        return "Balance Projection";
      default:
        return "View Chart";
    }
  };

  return (
    <div className="flex flex-wrap gap-2 p-4 border-b border-slate-800 bg-[#0D1117]/50 items-center justify-center z-10">
      {quickActionCharts.map((chartId) => (
        <button
          key={chartId}
          onClick={() => onTriggerChart(chartId)}
          disabled={isGenerating}
          className="bg-[#161B22] border border-slate-700 hover:border-[#00FFAA] text-slate-300 hover:text-[#00FFAA] transition-all px-4 py-2 rounded-xl text-xs font-bold disabled:opacity-50"
        >
          {getChartDisplayName(chartId)}
        </button>
      ))}
    </div>
  );
};
