"use client";

import React, { useState } from "react";
import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ContextMeterProps {
  tokens: number;
  limit?: number;
  className?: string;
}

export function ContextMeter({ tokens = 0, limit = 128000, className }: ContextMeterProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const safeTokens = Math.max(0, tokens || 0);
  const percent = Math.min(100, Math.round((safeTokens / limit) * 100));
  const isHigh = percent >= 85;
  const isMedium = percent >= 65 && percent < 85;

  const formattedUsed =
    safeTokens >= 1000 ? `${(safeTokens / 1000).toFixed(1)}k` : `${safeTokens}`;
  const formattedLimit =
    limit >= 1000 ? `${(limit / 1000).toFixed(0)}k` : `${limit}`;

  return (
    <div
      className={cn("relative inline-flex items-center", className)}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div
        className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 hover:border-white/20 transition-all cursor-pointer outline-none select-none"
      >
        <Cpu
          size={13}
          className={
            isHigh
              ? "text-danger animate-pulse"
              : isMedium
              ? "text-amber-400"
              : "text-primary/70"
          }
        />
        <div className="flex items-center gap-1.5 text-[11px] font-mono">
          <span className="text-muted-foreground/80">Context</span>
          <span className="text-foreground font-semibold">
            {formattedUsed}/{formattedLimit}
          </span>
          <span
            className={cn(
              "font-bold",
              isHigh
                ? "text-danger"
                : isMedium
                ? "text-amber-400"
                : "text-primary"
            )}
          >
            ({percent}%)
          </span>
        </div>
      </div>

      {showTooltip && (
        <div className="absolute right-0 top-full mt-2 w-64 p-3 rounded-2xl bg-[#0d0e14]/95 backdrop-blur-2xl border border-white/15 text-xs shadow-2xl z-50 text-foreground animate-in fade-in zoom-in-95 duration-150">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-foreground">Model Context Window</span>
              <span className="font-mono text-primary font-bold">{percent}%</span>
            </div>

            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-300",
                  isHigh ? "bg-danger" : isMedium ? "bg-amber-400" : "bg-primary"
                )}
                style={{ width: `${Math.max(2, percent)}%` }}
              />
            </div>

            <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
              <span>{safeTokens.toLocaleString()} tokens used</span>
              <span>{limit.toLocaleString()} max</span>
            </div>

            <p className="text-[10px] text-muted-foreground/80 leading-relaxed border-t border-white/5 pt-1.5 mt-0.5">
              Reflects active prompt and conversation memory allocated for this advisory session.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
