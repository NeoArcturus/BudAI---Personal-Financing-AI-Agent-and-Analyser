"use client";

import React from "react";
import { Newspaper, Globe2, TrendingUp, BarChart4 } from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex flex-col h-full w-full p-8 overflow-y-auto scrollbar-hide gap-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Market Overview</h1>
        <p className="text-slate-400">
          Latest financial intelligence and global market trends.
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col h-80">
          <div className="flex items-center gap-2 mb-4 text-[#69F0AE] font-bold">
            <Newspaper size={20} /> Financial & Market News
          </div>
          <div className="flex-1 flex items-center justify-center border border-dashed border-[#1A2D21] rounded-xl text-slate-500">
            [News API Feed Integrates Here]
          </div>
        </div>
        <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col h-80">
          <div className="flex items-center gap-2 mb-4 text-[#69F0AE] font-bold">
            <Globe2 size={20} /> Geo-Political Updates
          </div>
          <div className="flex-1 flex items-center justify-center border border-dashed border-[#1A2D21] rounded-xl text-slate-500">
            [Geo-Political API Feed Integrates Here]
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col h-96">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-[#69F0AE] font-bold">
              <TrendingUp size={20} /> Commodities Tracker
            </div>
            <div className="flex gap-2">
              <span className="px-3 py-1 bg-[#1A2D21] rounded-lg text-xs font-bold text-white cursor-pointer hover:bg-[#69F0AE]/20 hover:text-[#69F0AE]">
                OIL
              </span>
              <span className="px-3 py-1 bg-[#1A2D21] rounded-lg text-xs font-bold text-white cursor-pointer hover:bg-[#69F0AE]/20 hover:text-[#69F0AE]">
                GOLD
              </span>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center border border-dashed border-[#1A2D21] rounded-xl text-slate-500">
            <BarChart4 size={48} className="opacity-20 mb-4" />
          </div>
        </div>
        <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-6 flex flex-col h-96">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-[#69F0AE] font-bold">
              <TrendingUp size={20} /> Equities Tracker
            </div>
            <div className="flex gap-2">
              <span className="px-3 py-1 bg-[#1A2D21] rounded-lg text-xs font-bold text-white cursor-pointer hover:bg-[#69F0AE]/20 hover:text-[#69F0AE]">
                AAPL
              </span>
              <span className="px-3 py-1 bg-[#1A2D21] rounded-lg text-xs font-bold text-white cursor-pointer hover:bg-[#69F0AE]/20 hover:text-[#69F0AE]">
                TSLA
              </span>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center border border-dashed border-[#1A2D21] rounded-xl text-slate-500">
            <BarChart4 size={48} className="opacity-20 mb-4" />
          </div>
        </div>
      </div>
    </div>
  );
}
