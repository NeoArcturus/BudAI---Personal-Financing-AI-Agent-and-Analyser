"use client";

import React from "react";
import { Tooltip } from "@heroui/react";
import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ContextMeterProps {
  tokens: number;
  limit?: number;
  className?: string;
}

export function ContextMeter({ tokens, limit = 128000, className }: ContextMeterProps) {
  if (!tokens || tokens <= 0) return null;

  const percent = Math.min(100, Math.round((tokens / limit) * 100));
  const isHigh = percent >= 85;
  const isMedium = percent >= 65 && percent < 85;

  const formattedUsed = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : `${tokens}`;
  const formattedLimit = limit >= 1000 ? `${(limit / 1000).toFixed(0)}k` : `${limit}`;

  return (
    <Tooltip delay={100} closeDelay={150}>
      <Tooltip.Trigger className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-foreground/5 border border-white/10 hover:border-white/20 transition-all cursor-pointer outline-none select-none">
        <Cpu size={12} className={isHigh ? "text-danger" : isMedium ? "text-amber-400" : "text-muted-foreground"} />
        <span className="font-mono text-[10px] text-muted-foreground">
          {formattedUsed} / {formattedLimit}
        </span>
        <span
          className={cn(
            "font-mono text-[10px] font-bold",
            isHigh ? "text-danger" : isMedium ? "text-amber-400" : "text-primary"
          )}
        >
          ({percent}%)
        </span>
      </Tooltip.Trigger>
      <Tooltip.Content
        showArrow
        offset={6}
        className="w-56 p-2.5 rounded-xl bg-[#0b0c14]/90 backdrop-blur-xl border border-white/20 text-xs shadow-xl text-foreground"
      >
        <Tooltip.Arrow className="fill-[#0b0c14]/90 stroke-white/20" />
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] font-semibold">
            <span>Session Memory</span>
            <span className="font-mono text-primary">{percent}% used</span>
          </div>
          <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-300",
                isHigh ? "bg-danger" : isMedium ? "bg-amber-400" : "bg-primary"
              )}
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="text-[10px] text-muted-foreground leading-relaxed mt-0.5">
            {tokens.toLocaleString()} of {limit.toLocaleString()} tokens used in this session.
          </p>
        </div>
      </Tooltip.Content>
    </Tooltip>
  );
}
