"use client";

import React from "react";
import CashFlowWidgetClient from "../../_components/widgets/Cashflow/client";
import ExpenseDistributionWidgetClient from "../../_components/widgets/Categories/client";
import LedgerTableWidgetClient from "../../_components/widgets/Transactions/client";

export interface DynamicChartConfig {
  type: string;
  title: string;
  subtitle?: string;
  labels?: string[];
  datasets?: any[];
  options?: any;
}

export const DynamicChart = ({ config, hideWrapper = false }: { config: DynamicChartConfig, hideWrapper?: boolean }) => {
  const normalizedType = (config.type || "bar").replace("_chart", "").toLowerCase();
  const normalizedTitle = (config.title || "").toLowerCase();

  if (normalizedType === "pie" || normalizedType === "doughnut" || normalizedTitle.includes("categor")) {
    return (
      <div className="w-full h-full relative">
         <ExpenseDistributionWidgetClient />
      </div>
    );
  }

  if (normalizedType === "table" || normalizedType === "transactions" || normalizedTitle.includes("transaction")) {
    return (
      <div className="w-full h-full relative">
         <LedgerTableWidgetClient />
      </div>
    );
  }

  // Default to Cashflow
  return (
    <div className="w-full h-full relative">
       <CashFlowWidgetClient />
    </div>
  );
};
