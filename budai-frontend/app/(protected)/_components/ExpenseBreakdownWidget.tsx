// app/(protected)/_components/ExpenseBreakdownWidget.tsx
import React from "react";
import { Sparkles, MoreVertical } from "lucide-react";
import CoreChartEngine from "./CoreChartEngine";
import { NativeChartConfig, ParsedCategory } from "@/types";
import { Spinner } from "@heroui/react";

interface ExpenseBreakdownWidgetProps {
  isLoading: boolean;
  config: NativeChartConfig | null;
  categories: ParsedCategory[];
  onAnalyze: () => void;
}

export default function ExpenseBreakdownWidget({
  isLoading,
  config,
  categories,
  onAnalyze,
}: ExpenseBreakdownWidgetProps) {
  const getCategoryTheme = (index: number) => {
    const colors = [
      "bg-[#FF8A4C]",
      "bg-[#3D73FF]",
      "bg-[#FF5E98]",
      "bg-[#00E096]",
    ];
    return colors[index % colors.length];
  };

  let currentMonthTotal = 0;
  let lastMonthTotal = 0;
  let dataArr: number[] = [];

  if (
    config &&
    config.data &&
    config.data.datasets &&
    config.data.datasets.length > 0
  ) {
    dataArr = config.data.datasets[0].data as number[];
    if (dataArr.length >= 1)
      currentMonthTotal = dataArr[dataArr.length - 1] || 0;
    if (dataArr.length >= 2) lastMonthTotal = dataArr[dataArr.length - 2] || 0;
  }

  const diff = currentMonthTotal - lastMonthTotal;
  const percentChange = lastMonthTotal > 0 ? (diff / lastMonthTotal) * 100 : 0;
  const isUp = diff > 0;
  const hasLastMonthData = dataArr.length >= 2;

  return (
    <div className="w-full h-full bg-[#13151D]/40 bg-linear-to-br from-white/8 to-transparent backdrop-blur-xl rounded-3xl border border-white/8 p-6 flex flex-col shadow-2xl">
      <div className="flex justify-between items-start mb-6 shrink-0">
        <div>
          <h3 className="text-white font-semibold text-lg tracking-tight mb-1">
            Expenses Breakdown
          </h3>
          <p className="text-xs text-[#8B8E98]">This month</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onAnalyze}
            className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-[#8B8E98] hover:text-[#3D73FF] hover:bg-[#3D73FF]/10 transition-colors"
          >
            <Sparkles size={14} />
          </button>
          <button className="text-[#8B8E98] hover:text-white transition-colors">
            <MoreVertical size={20} />
          </button>
        </div>
      </div>

      <div className="mb-4 shrink-0">
        <h4 className="text-3xl font-bold text-white tracking-tight mb-1">
          £{" "}
          {currentMonthTotal.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}
        </h4>
        <p className="text-xs font-medium text-[#5E6272]">
          Compared to last month:{" "}
          <span
            className={
              hasLastMonthData
                ? isUp
                  ? "text-[#FF8A4C]"
                  : "text-[#00E096]"
                : "text-[#5E6272]"
            }
          >
            {hasLastMonthData
              ? `${isUp ? "+" : ""}${percentChange.toFixed(1)}%`
              : "N/A"}
          </span>
        </p>
      </div>

      <div className="flex-1 min-h-37.5 max-h-50 border-b border-white/10 pb-4 mb-4 flex items-center justify-center">
        {isLoading ? (
          <Spinner color="accent" />
        ) : config ? (
          <CoreChartEngine config={config} />
        ) : (
          <span className="text-[#5E6272] text-sm font-medium">
            No Data Available
          </span>
        )}
      </div>

      <div className="space-y-4 shrink-0 overflow-y-auto scrollbar-hide">
        {categories.slice(0, 4).map((c, idx) => (
          <div key={idx} className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div
                className={`w-3 h-3 rounded-md ${getCategoryTheme(idx)}`}
              ></div>
              <span className="text-sm text-[#8B8E98] font-medium">
                {c.name}
              </span>
            </div>
            <span className="text-sm font-semibold text-white">
              £{" "}
              {c.value.toLocaleString(undefined, {
                minimumFractionDigits: 0,
              })}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
