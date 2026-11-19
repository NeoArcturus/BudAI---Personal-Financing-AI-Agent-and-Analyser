"use client";

import React from "react";
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
  if (!steps.length) return null;

  const hasRunning = steps.some((s) => s.status === "running" || s.status === "queued");
  const hasFailed = steps.some((s) => s.status === "failed");
  const completedCount = steps.filter((s) => s.status === "completed").length;

  const headerTitle =
    title ||
    (hasRunning
      ? `Running checks (${completedCount}/${steps.length})...`
      : hasFailed
      ? `Checks completed with errors (${completedCount}/${steps.length})`
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
              const duration = step.durationMs
                ? `${(step.durationMs / 1000).toFixed(1)}s`
                : undefined;

              return (
                <TraceLine
                  key={step.id || idx}
                  status={isRunning ? "running" : isFailed ? "error" : "completed"}
                  className="pl-2"
                >
                  <TraceTitle className="text-[11px] text-muted-foreground">
                    {step.action || step.name}
                  </TraceTitle>
                  {duration && <TraceDescription>{duration}</TraceDescription>}
                </TraceLine>
              );
            })}
          </div>
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
