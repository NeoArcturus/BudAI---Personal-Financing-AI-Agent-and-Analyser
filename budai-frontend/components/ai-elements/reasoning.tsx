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
              <Brain size={13} className={isStreaming ? "text-primary" : "text-muted-foreground"} />
            </TraceIcon>
            <TraceTitle>{title}</TraceTitle>
            {secondsDisplay && <TraceDescription>{secondsDisplay}</TraceDescription>}
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent className="whitespace-pre-wrap font-mono text-[11px] text-muted-foreground/90 max-h-60 overflow-y-auto scrollbar-hide">
          {reasoningText}
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
