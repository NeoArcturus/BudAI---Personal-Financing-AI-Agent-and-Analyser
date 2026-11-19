"use client";

import React from "react";
import CashFlowWidgetClient from "@/app/(protected)/_components/widgets/Cashflow/client";
import ExpenseDistributionWidgetClient from "@/app/(protected)/_components/widgets/Categories/client";
import TransactionsWidgetClient from "@/app/(protected)/_components/widgets/Transactions/client";

export interface DynamicChartConfig {
  type: string;
  title: string;
  subtitle?: string;
  labels?: string[];
  datasets?: any[];
  options?: any;
}

export const DynamicChart = ({
  config,
}: {
  config: DynamicChartConfig;
  hideWrapper?: boolean;
}) => {
  const normalizedType = (config.type || "bar").replace("_chart", "").toLowerCase();
  const normalizedTitle = (config.title || "").toLowerCase();

  if (
    normalizedType === "pie" ||
    normalizedType === "doughnut" ||
    normalizedTitle.includes("categor")
  ) {
    return (
      <div className="w-full h-full relative my-2 rounded-2xl overflow-hidden border border-white/10 bg-[#0d0e14]/40 backdrop-blur-xl p-3">
        <ExpenseDistributionWidgetClient />
      </div>
    );
  }

  if (
    normalizedType === "table" ||
    normalizedType === "transactions" ||
    normalizedTitle.includes("transaction")
  ) {
    return (
      <div className="w-full h-full relative my-2 rounded-2xl overflow-hidden border border-white/10 bg-[#0d0e14]/40 backdrop-blur-xl p-3">
        <TransactionsWidgetClient />
      </div>
    );
  }

  // Default to Cashflow
  return (
    <div className="w-full h-full relative my-2 rounded-2xl overflow-hidden border border-white/10 bg-[#0d0e14]/40 backdrop-blur-xl p-3">
      <CashFlowWidgetClient />
    </div>
  );
};
