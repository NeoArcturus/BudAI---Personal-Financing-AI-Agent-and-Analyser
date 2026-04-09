"use client";

import React from "react";
import { TabType } from "@/types";
import { Button } from "@/components/ui/button";

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
  if (activeAccountId === "ALL" || !activeAccountId) return null;

  const quickActionCharts = [
    { id: "historical_monthly", label: "Monthly History" },
    { id: "categorized", label: "Category Breakdown" },
    { id: "cash_flow_mixed", label: "Income vs Expenses" },
    { id: "health_radar", label: "Health Radar" },
    { id: "expense_forecast", label: "Expense Forecast" },
    { id: "balance_forecast", label: "Balance Projection" },
  ];

  return (
    <div className="flex flex-wrap gap-2 p-4 border-b border-slate-800 bg-[#0D1117]/50 items-center justify-center z-10">
      {quickActionCharts.map((chart) => (
        <Button
          key={chart.id}
          variant="outline"
          onClick={() => onTriggerChart(chart.id)}
          disabled={isGenerating}
          className="bg-[#161B22] border-slate-700 hover:border-[#00FFAA] text-slate-300 hover:text-[#00FFAA] hover:bg-[#00FFAA]/10 transition-all text-[10px] font-bold uppercase tracking-widest"
        >
          {chart.label}
        </Button>
      ))}
    </div>
  );
};
