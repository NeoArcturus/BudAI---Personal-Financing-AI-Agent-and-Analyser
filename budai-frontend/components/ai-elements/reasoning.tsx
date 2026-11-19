"use client";

import React, { useState, useEffect } from "react";
import { Brain } from "lucide-react";
import {
  Trace,
  TraceCollapsible,
  TraceCollapsibleTrigger,
  TraceCollapsibleContent,
  TraceLine,
  TraceIcon,
  TraceTitle,
  TraceDescription,
} from "./trace";

export interface ReasoningProps {
  reasoningText: string;
  isStreaming?: boolean;
  durationMs?: number;
  defaultOpen?: boolean;
  className?: string;
}

export function Reasoning({
  reasoningText,
  isStreaming = false,
  durationMs,
  defaultOpen = false,
  className,
}: ReasoningProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [elapsedSec, setElapsedSec] = useState(0);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isStreaming) {
      const start = Date.now();
      timer = setInterval(() => {
        setElapsedSec(Math.max(1, Math.floor((Date.now() - start) / 1000)));
      }, 500);
    }
    return () => clearInterval(timer);
  }, [isStreaming]);

  if (!reasoningText && !isStreaming) return null;

  const secondsDisplay = durationMs
    ? `${(durationMs / 1000).toFixed(1)}s`
    : elapsedSec > 0
    ? `${elapsedSec}s`
    : undefined;

  const title = isStreaming ? "Thinking..." : "Thought";

  return (
    <Trace className={className}>
      <TraceCollapsible
        open={open}
        defaultOpen={defaultOpen}
        onOpenChange={setOpen}
      >
        <TraceCollapsibleTrigger>
          <TraceLine status={isStreaming ? "running" : "completed"}>
            <TraceIcon>
              <Brain size={14} className={isStreaming ? "text-primary animate-pulse" : "text-primary/70"} />
            </TraceIcon>
            <TraceTitle className="font-semibold text-foreground/90">{title}</TraceTitle>
            {secondsDisplay && (
              <TraceDescription className="text-muted-foreground font-mono text-[10px] bg-white/5 px-2 py-0.5 rounded-full border border-white/10">
                {secondsDisplay}
              </TraceDescription>
            )}
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent className="whitespace-pre-wrap font-sans text-xs text-muted-foreground/90 leading-relaxed max-h-64 overflow-y-auto scrollbar-hide border-l-2 border-primary/30 pl-3 py-1 my-1 bg-white/[0.01] rounded-r-lg">
          {reasoningText}
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
