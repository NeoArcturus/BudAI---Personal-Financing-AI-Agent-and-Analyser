"use client";

import React from "react";
import { Bookmark, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

export interface CheckpointProps {
  label: string;
  onRestore?: () => void;
  className?: string;
}

export function Checkpoint({ label, onRestore, className }: CheckpointProps) {
  return (
    <div
      className={cn(
        "w-full flex items-center justify-between py-2 px-3.5 my-3 rounded-xl bg-white/[0.02] border border-dashed border-white/15 text-xs text-muted-foreground",
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Bookmark size={13} className="text-primary/80 shrink-0" />
        <span className="font-medium text-foreground/80 text-[11px]">
          Save point: <span className="font-semibold text-foreground">{label}</span>
        </span>
      </div>

      {onRestore && (
        <button
          type="button"
          onClick={onRestore}
          className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-semibold text-primary hover:text-primary-foreground hover:bg-primary transition-all cursor-pointer border border-primary/30"
        >
          <RotateCcw size={10} />
          <span>Go back to this point</span>
        </button>
      )}
    </div>
  );
}
