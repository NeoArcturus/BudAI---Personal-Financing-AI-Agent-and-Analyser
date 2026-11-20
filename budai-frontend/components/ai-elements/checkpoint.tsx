"use client";

import React from "react";
import { Bookmark, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Marker,
  MarkerContent,
} from "@/components/ui/marker";
import { Button } from "@/components/ui/button";

export interface CheckpointProps {
  label: string;
  onRestore?: () => void;
  className?: string;
}

export function Checkpoint({ label, onRestore, className }: CheckpointProps) {
  return (
    <Marker
      variant="separator"
      className={cn("w-full my-4 text-xs font-mono select-none", className)}
    >
      <MarkerContent className="inline-flex items-center gap-2.5 px-3 py-1 rounded-full bg-[#121d2b]/90 border border-[#1f2d40] text-muted-foreground shadow-xs">
        <Bookmark size={12} className="text-primary/90 shrink-0" />
        <span className="text-[11px] font-medium text-foreground/85">
          Save point: <span className="font-semibold text-foreground">{label}</span>
        </span>
        {onRestore && (
          <Button
            size="xs"
            variant="outline"
            onClick={onRestore}
            className="h-5 px-2 text-[10px] font-semibold gap-1 text-primary border-primary/30 hover:bg-primary/10 rounded-full cursor-pointer ml-1"
          >
            <RotateCcw size={10} />
            <span>Restore</span>
          </Button>
        )}
      </MarkerContent>
    </Marker>
  );
}
