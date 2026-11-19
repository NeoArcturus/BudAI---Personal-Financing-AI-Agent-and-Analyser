"use client";

import React from "react";
import { ListTodo, ListChecks, Circle, CheckCircle2 } from "lucide-react";
import {
  Trace,
  TraceCollapsible,
  TraceCollapsibleTrigger,
  TraceCollapsibleContent,
  TraceLine,
  TraceIcon,
  TraceTitle,
} from "./trace";

export interface QueueItem {
  id: string;
  title: string;
  description?: string;
  status: "todo" | "completed";
}

export interface QueueProps {
  items: QueueItem[];
  defaultOpen?: boolean;
  className?: string;
}

export function Queue({ items, defaultOpen = true, className }: QueueProps) {
  if (!items.length) return null;

  const todoItems = items.filter((i) => i.status === "todo");
  const completedItems = items.filter((i) => i.status === "completed");

  return (
    <Trace className={className}>
      {todoItems.length > 0 && (
        <TraceCollapsible defaultOpen={defaultOpen}>
          <TraceCollapsibleTrigger>
            <TraceLine>
              <TraceIcon>
                <ListTodo size={13} className="text-primary" />
              </TraceIcon>
              <TraceTitle>
                {todoItems.length} {todoItems.length === 1 ? "item" : "items"} to do
              </TraceTitle>
            </TraceLine>
          </TraceCollapsibleTrigger>
          <TraceCollapsibleContent keepMounted>
            <div className="flex flex-col gap-1.5 pt-1">
              {todoItems.map((item) => (
                <div key={item.id} className="flex flex-col gap-0.5 pl-2">
                  <TraceLine className="text-[11px]">
                    <TraceIcon>
                      <Circle size={10} className="text-muted-foreground" />
                    </TraceIcon>
                    <TraceTitle>{item.title}</TraceTitle>
                  </TraceLine>
                  {item.description && (
                    <span className="text-[10px] text-muted-foreground/80 pl-5">
                      {item.description}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </TraceCollapsibleContent>
        </TraceCollapsible>
      )}

      {completedItems.length > 0 && (
        <TraceCollapsible defaultOpen={false}>
          <TraceCollapsibleTrigger>
            <TraceLine status="completed">
              <TraceIcon>
                <ListChecks size={13} className="text-emerald-400" />
              </TraceIcon>
              <TraceTitle>
                {completedItems.length} completed
              </TraceTitle>
            </TraceLine>
          </TraceCollapsibleTrigger>
          <TraceCollapsibleContent keepMounted>
            <div className="flex flex-col gap-1.5 pt-1">
              {completedItems.map((item) => (
                <div key={item.id} className="flex flex-col gap-0.5 pl-2">
                  <TraceLine className="text-[11px] text-muted-foreground line-through">
                    <TraceIcon>
                      <CheckCircle2 size={11} className="text-emerald-400" />
                    </TraceIcon>
                    <TraceTitle>{item.title}</TraceTitle>
                  </TraceLine>
                  {item.description && (
                    <span className="text-[10px] text-muted-foreground/60 pl-5">
                      {item.description}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </TraceCollapsibleContent>
        </TraceCollapsible>
      )}
    </Trace>
  );
}
