"use client";

import React, { useState, useEffect } from "react";
import { CheckCircle2, CircleDashed, AlertCircle, Layers } from "lucide-react";
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

export interface TaskStepItem {
  id?: string;
  name: string;
  action?: string;
  status: "queued" | "running" | "completed" | "failed";
  durationMs?: number;
}

export interface TaskProps {
  title?: string;
  steps: TaskStepItem[];
  defaultOpen?: boolean;
  className?: string;
}

export function Task({
  title,
  steps,
  defaultOpen = false,
  className,
}: TaskProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const hasRunning = steps.some((s) => s.status === "running" || s.status === "queued");
  const hasFailed = steps.some((s) => s.status === "failed");
  const completedCount = steps.filter((s) => s.status === "completed").length;

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (hasRunning) {
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [hasRunning]);

  if (!steps.length) return null;

  const headerTitle =
    title ||
    (hasRunning
      ? `Checking records (${completedCount}/${steps.length}) ${elapsedSeconds > 0 ? `(${elapsedSeconds}s)` : ""}...`
      : hasFailed
      ? `Checks completed with issues (${completedCount}/${steps.length})`
      : `Checks completed (${steps.length} steps)`);

  return (
    <Trace className={className}>
      <TraceCollapsible defaultOpen={defaultOpen}>
        <TraceCollapsibleTrigger>
          <TraceLine
            status={hasRunning ? "running" : hasFailed ? "error" : "completed"}
          >
            <TraceTitle className="font-normal text-muted-foreground group-hover:text-foreground text-xs">
              {headerTitle}
            </TraceTitle>
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent keepMounted>
          <div className="flex flex-col gap-1.5 pt-1">
            {steps.map((step, idx) => {
              const isRunning = step.status === "running" || step.status === "queued";
              const isFailed = step.status === "failed";
              const duration = isRunning && elapsedSeconds > 0
                ? `${elapsedSeconds}s`
                : step.durationMs
                ? `${(step.durationMs / 1000).toFixed(1)}s`
                : undefined;

              return (
                <div key={step.id || idx} className="flex flex-col">
                  <TraceLine
                    status={isRunning ? "running" : isFailed ? "error" : "completed"}
                    className="pl-2"
                  >
                    <TraceTitle className="text-[11px] text-muted-foreground">
                      {step.action || step.name}
                    </TraceTitle>
                    {duration && <TraceDescription>{duration}</TraceDescription>}
                  </TraceLine>
                  {isRunning && elapsedSeconds >= 20 && (
                    <span className="text-[10px] text-primary/75 italic pl-6 pt-0.5 animate-pulse">
                      Gathering records across all your connected accounts...
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
