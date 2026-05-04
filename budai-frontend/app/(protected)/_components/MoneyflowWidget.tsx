import React from "react";
import { MoreVertical } from "lucide-react";
import CoreChartEngine from "./CoreChartEngine";
import { NativeChartConfig } from "@/types";
import { Spinner } from "@heroui/react";

interface MoneyFlowWidgetProps {
  isLoading: boolean;
  config: NativeChartConfig | null;
}

export default function MoneyFlowWidget({
  isLoading,
  config,
}: MoneyFlowWidgetProps) {
  return (
    <div className="w-full h-full bg-[#13151D]/40 bg-linear-to-br from-white/8 to-transparent backdrop-blur-xl rounded-3xl border border-white/8 p-6 flex flex-col shadow-2xl">
      <div className="flex justify-between items-center mb-6">
        <div className="flex flex-col gap-2">
          <h3 className="text-white font-semibold text-lg tracking-tight">
            Money Flow
          </h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[#3D73FF]"></div>
              <span className="text-xs text-[#8B8E98] font-medium">Income</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[#FF8A4C]"></div>
              <span className="text-xs text-[#8B8E98] font-medium">
                Outcome
              </span>
            </div>
          </div>
        </div>
        <button className="text-[#8B8E98] hover:text-white transition-colors">
          <MoreVertical size={20} />
        </button>
      </div>

      <div className="flex-1 min-h-50 w-full flex items-center justify-center">
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
    </div>
  );
}
