"use client";

import React from "react";
import { Card, Chip, Button, ProgressBar } from "@heroui/react";
import { Plus } from "lucide-react";

export interface BucketItem {
  id: string;
  name: string;
  type: string;
  cached_balance: number;
  target_amount?: number | null;
  priority_index: number;
  allocation_percentage: number;
  daily_burn?: number;
  status_note?: string;
}

interface BucketAllocationsPanelProps {
  buckets: BucketItem[];
  onOpenTransferModal?: () => void;
}

export function BucketAllocationsPanel({
  buckets,
  onOpenTransferModal,
}: BucketAllocationsPanelProps) {
  const getFillColor = (type: string) => {
    switch (type) {
      case "DEFAULT":
        return "bg-cyan-500";
      case "RECURRING":
        return "bg-gradient-to-r from-cyan-400 to-blue-500";
      case "ACCUMULATING":
        return "bg-emerald-500";
      case "TARGET_DATE":
        return "bg-indigo-500";
      case "LIABILITY":
        return "bg-amber-500";
      default:
        return "bg-slate-500";
    }
  };

  return (
    <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none space-y-4">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-1 w-full">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Capital Allocations
          </h2>
          <p className="text-xs text-slate-500">5 Dynamic Smart Envelopes</p>
        </div>
        <Chip size="sm" variant="soft" color="default" className="font-mono text-xs">
          100% Allocated
        </Chip>
      </div>

      {/* Bucket Cards */}
      <div className="space-y-2.5 w-full">
        {buckets.map((b) => {
          const isRecurring = b.type === "RECURRING";
          return (
            <Card
              key={b.id}
              className={`p-3 rounded-xl border transition shadow-none ${
                isRecurring
                  ? "bg-slate-900/90 border-cyan-500/40 neon-glow-cyan"
                  : "bg-slate-900/60 border-slate-800 glass-card-hover"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`text-xs font-medium flex items-center gap-1.5 ${
                    isRecurring ? "text-cyan-300 font-semibold" : "text-slate-200"
                  }`}
                >
                  {isRecurring && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                  {b.name}
                </span>
                <span
                  className={`text-xs font-mono ${
                    isRecurring ? "text-cyan-400/80" : "text-slate-400"
                  }`}
                >
                  Priority {Math.round(b.priority_index)}
                </span>
              </div>

              <div className="flex items-baseline justify-between mb-2">
                <span className="text-lg font-bold text-white tracking-tight font-mono tabular-nums">
                  £{b.cached_balance.toLocaleString("en-GB", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  {b.allocation_percentage}% of total
                </span>
              </div>

              {/* HeroUI ProgressBar */}
              <ProgressBar
                aria-label={`Allocation for ${b.name}`}
                value={Math.min(100, Math.max(5, b.allocation_percentage))}
                className="w-full"
              >
                <ProgressBar.Track className="bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <ProgressBar.Fill className={`h-full rounded-full ${getFillColor(b.type)}`} />
                </ProgressBar.Track>
              </ProgressBar>

              {/* Metadata note */}
              <div className="flex justify-between items-center mt-2 text-[11px] text-slate-400 font-mono">
                <span>
                  {b.daily_burn && b.daily_burn > 0
                    ? `Daily burn ~£${b.daily_burn.toFixed(0)}/day`
                    : b.target_amount
                    ? `Target: £${b.target_amount.toLocaleString()}`
                    : "Autopay scheduled"}
                </span>
                <span
                  className={
                    isRecurring
                      ? "text-cyan-400 font-medium"
                      : b.type === "DEFAULT"
                      ? "text-emerald-400"
                      : "text-slate-400"
                  }
                >
                  {b.status_note || "Active"}
                </span>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Manual Transfer Action Button */}
      <Button
        variant="secondary"
        onPress={onOpenTransferModal}
        className="w-full mt-2 py-2 px-3 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-xs font-medium text-slate-300 border border-slate-700 flex items-center justify-center space-x-2 transition"
      >
        <Plus className="w-3.5 h-3.5 text-cyan-400" />
        <span>Reallocate or Adjust Policy</span>
      </Button>
    </Card>
  );
}
