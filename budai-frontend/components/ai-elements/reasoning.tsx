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
            <TraceTitle className="font-normal text-muted-foreground group-hover:text-foreground text-xs">
              {title}
            </TraceTitle>
            {secondsDisplay && (
              <TraceDescription className="text-muted-foreground/60 text-[10px] ml-1">
                {secondsDisplay}
              </TraceDescription>
            )}
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent className="whitespace-pre-wrap font-sans text-xs text-muted-foreground/85 leading-relaxed max-h-64 overflow-y-auto scrollbar-hide py-1">
          {reasoningText}
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
