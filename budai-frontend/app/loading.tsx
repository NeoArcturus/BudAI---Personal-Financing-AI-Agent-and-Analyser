"use client";

import React from "react";
import { Activity } from "lucide-react";

export default function Loading() {
  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center p-6 text-center relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-125 h-125 bg-primary/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="flex flex-col items-center gap-6 relative z-10">
        <div className="relative">
          <Activity className="text-primary w-16 h-16 animate-pulse" />
          <div className="absolute inset-0 bg-primary/20 blur-xl animate-pulse rounded-full" />
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <h2 className="text-xl font-bold text-foreground tracking-[0.2em] uppercase italic">
              Syncing Intelligence
            </h2>
            <div className="w-48 h-1 bg-primary/10 rounded-full overflow-hidden mx-auto">
              <div className="h-full bg-primary w-1/3 animate-[marquee_2s_linear_infinite]" />
            </div>
          </div>
          <p className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em]">
            Calculating Wealth Velocity
          </p>
        </div>
      </div>
    </div>
  );
}
