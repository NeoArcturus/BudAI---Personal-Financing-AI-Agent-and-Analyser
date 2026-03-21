// app/dashboard/_components/SidebarRight.tsx
"use client";

import React from "react";
import { Globe } from "lucide-react";

export const SidebarRight: React.FC = () => {
  return (
    <aside className="w-[22%] p-6 bg-[#161B22] border-l border-slate-800 overflow-y-auto shrink-0 scrollbar-hide">
      <h2 className="text-xs font-bold tracking-widest text-slate-500 mb-6 flex items-center gap-2">
        <Globe className="w-4 h-4" /> GLOBAL MARKETS
      </h2>
      <div className="space-y-4">
        {[1, 2, 3].map((item) => (
          <div
            key={item}
            className="bg-[#0D1117] p-5 rounded-2xl border border-slate-800"
          >
            <div className="h-4 w-1/4 bg-slate-800 rounded mb-3"></div>
            <div className="h-4 w-full bg-slate-800 rounded mb-2"></div>
            <div className="h-4 w-5/6 bg-slate-800 rounded"></div>
            <p className="text-[10px] text-slate-500 mt-4 uppercase">
              Geopolitical Event
            </p>
          </div>
        ))}
      </div>
    </aside>
  );
};
