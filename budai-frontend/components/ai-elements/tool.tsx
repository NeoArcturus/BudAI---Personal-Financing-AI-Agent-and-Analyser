"use client";

import React from "react";
import { Wrench } from "lucide-react";
import {
  Trace,
  TraceCollapsible,
  TraceCollapsibleTrigger,
  TraceCollapsibleContent,
  TraceLine,
  TraceIcon,
  TraceTitle,
} from "./trace";

export interface ToolInvocationProps {
  toolName: string;
  args?: Record<string, unknown>;
  state?: "partial-call" | "call" | "result" | string;
  result?: unknown;
  className?: string;
}

export function ToolInvocation({
  toolName,
  args,
  state = "result",
  result,
  className,
}: ToolInvocationProps) {
  const isRunning = state === "partial-call" || state === "call";
  const isError = state === "error";

  // Friendly display names in plain English
  const friendlyName =
    toolName === "split_transaction"
      ? "Split transaction"
      : toolName === "reorder_dashboard_widgets"
      ? "Reorder dashboard"
      : toolName === "generate_ui_chart"
      ? "Generate financial chart"
      : toolName === "execute_tool"
      ? "Execute financial check"
      : toolName;

  const title = isRunning
    ? `Running ${friendlyName}...`
    : isError
    ? `Failed to run ${friendlyName}`
    : `Ran ${friendlyName}`;

  return (
    <Trace className={className}>
      <TraceCollapsible defaultOpen={false}>
        <TraceCollapsibleTrigger>
          <TraceLine status={isRunning ? "running" : isError ? "error" : "completed"}>
            <TraceIcon>
              <Wrench size={13} className="text-primary" />
            </TraceIcon>
            <TraceTitle>{title}</TraceTitle>
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent>
          <div className="flex flex-col gap-2 pt-1 font-mono text-[10px]">
            {args && Object.keys(args).length > 0 && (
              <div>
                <span className="text-muted-foreground/70 uppercase tracking-wider block mb-0.5">
                  Parameters:
                </span>
                <pre className="p-2 rounded-lg bg-black/40 border border-white/5 overflow-x-auto text-foreground/80 scrollbar-hide">
                  {JSON.stringify(args, null, 2)}
                </pre>
              </div>
            )}
            {result !== undefined && (
              <div>
                <span className="text-muted-foreground/70 uppercase tracking-wider block mb-0.5">
                  Output:
                </span>
                <pre className="p-2 rounded-lg bg-black/40 border border-white/5 overflow-x-auto text-emerald-400/90 scrollbar-hide">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </TraceCollapsibleContent>
      </TraceCollapsible>
    </Trace>
  );
}
