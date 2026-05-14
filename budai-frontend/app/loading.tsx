"use client";

import React from "react";
import { Activity } from "lucide-react";

export default function Loading() {
  return (
    <div className="min-h-screen bg-[#08090D] flex flex-col items-center justify-center p-6 text-center relative overflow-hidden bg-grid-pattern">
      {/* Ambient Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />
      
      <div className="flex flex-col items-center gap-6 relative z-10">
        <div className="relative">
          <Activity className="text-cyan-400 w-16 h-16 animate-pulse" />
          <div className="absolute inset-0 bg-cyan-400/20 blur-xl animate-pulse rounded-full" />
        </div>
        
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white tracking-[0.2em] uppercase">Syncing Intelligence</h2>
          <div className="w-48 h-1 bg-white/5 rounded-full overflow-hidden mx-auto">
            <div className="h-full bg-cyan-400 w-1/3 animate-[marquee_2s_linear_infinite]" />
          </div>
          <p className="text-white/30 text-xs font-mono mt-4">CALCULATING WEALTH VELOCITY...</p>
        </div>
      </div>
    </div>
  );
}
