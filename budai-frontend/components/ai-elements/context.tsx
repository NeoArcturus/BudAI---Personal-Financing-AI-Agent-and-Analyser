"use client";

import React, { createContext, useContext, useMemo } from "react";
import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

export interface ContextUsageData {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
}

export interface ContextValue {
  usedTokens: number;
  maxTokens: number;
  usage?: ContextUsageData;
  modelId?: string;
  percent: number;
}

const ContextContext = createContext<ContextValue | null>(null);

export function useModelContext(): ContextValue {
  const ctx = useContext(ContextContext);
  if (!ctx) {
    return {
      usedTokens: 0,
      maxTokens: 20224,
      percent: 0,
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    };
  }
  return ctx;
}

export interface ContextProps extends React.ComponentProps<typeof HoverCard> {
  usedTokens?: number;
  maxTokens?: number;
  usage?: ContextUsageData;
  modelId?: string;
}

export function Context({
  usedTokens = 0,
  maxTokens = 20224,
  usage,
  modelId = "qwen3.5-9b",
  children,
  ...props
}: ContextProps) {
  const safeUsed = Math.max(0, usedTokens || usage?.totalTokens || 0);
  const safeMax = maxTokens > 0 ? maxTokens : 20224;
  const percent = Math.min(100, Math.round((safeUsed / safeMax) * 100));

  const resolvedUsage: ContextUsageData = useMemo(() => {
    return {
      inputTokens: usage?.inputTokens ?? Math.max(0, safeUsed - (usage?.outputTokens ?? 0)),
      outputTokens: usage?.outputTokens ?? 0,
      totalTokens: safeUsed,
      reasoningTokens: usage?.reasoningTokens ?? 0,
    };
  }, [usage, safeUsed]);

  const value = useMemo(
    () => ({
      usedTokens: safeUsed,
      maxTokens: safeMax,
      usage: resolvedUsage,
      modelId,
      percent,
    }),
    [safeUsed, safeMax, resolvedUsage, modelId, percent]
  );

  return (
    <ContextContext.Provider value={value}>
      <HoverCard {...props}>{children}</HoverCard>
    </ContextContext.Provider>
  );
}

export interface ContextTriggerProps
  extends React.ComponentProps<typeof HoverCardTrigger> {
  children?: React.ReactNode;
}

export function ContextTrigger({
  className,
  children,
  ...props
}: ContextTriggerProps) {
  const { usedTokens, maxTokens, percent } = useModelContext();
  const isHigh = percent >= 85;
  const isMedium = percent >= 65 && percent < 85;

  const formattedUsed =
    usedTokens >= 1000 ? `${(usedTokens / 1000).toFixed(1)}k` : `${usedTokens}`;
  const formattedLimit =
    maxTokens >= 1000 ? `${(maxTokens / 1000).toFixed(0)}k` : `${maxTokens}`;

  return (
    <HoverCardTrigger
      className={cn("cursor-pointer outline-none", className)}
      {...props}
    >
      {children || (
        <Button
          variant="outline"
          size="sm"
          className="h-8 gap-2 px-2.5 rounded-xl border-white/10 bg-white/[0.04] hover:bg-white/[0.08] text-xs font-mono transition-all"
        >
          <Cpu
            size={13}
            className={cn(
              isHigh
                ? "text-danger animate-pulse"
                : isMedium
                ? "text-amber-400"
                : "text-primary/70"
            )}
          />
          <span className="text-muted-foreground/80">Context</span>
          <span className="text-foreground font-semibold">
            {formattedUsed}/{formattedLimit}
          </span>
          <span
            className={cn(
              "font-bold",
              isHigh
                ? "text-danger"
                : isMedium
                ? "text-amber-400"
                : "text-primary"
            )}
          >
            ({percent}%)
          </span>
        </Button>
      )}
    </HoverCardTrigger>
  );
}

export interface ContextContentProps
  extends React.ComponentProps<typeof HoverCardContent> {}

export function ContextContent({
  className,
  children,
  ...props
}: ContextContentProps) {
  return (
    <HoverCardContent
      className={cn(
        "w-72 p-4 flex flex-col gap-3 rounded-2xl bg-[#0d0e14]/95 backdrop-blur-2xl border border-white/15 text-xs text-foreground shadow-2xl",
        className
      )}
      {...props}
    >
      {children}
    </HoverCardContent>
  );
}

export interface ContextContentHeaderProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export function ContextContentHeader({
  className,
  ...props
}: ContextContentHeaderProps) {
  const { usedTokens, maxTokens, percent } = useModelContext();
  const isHigh = percent >= 85;
  const isMedium = percent >= 65 && percent < 85;

  return (
    <div className={cn("flex flex-col gap-2", className)} {...props}>
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-foreground">Model Context</span>
        <span
          className={cn(
            "font-mono font-bold text-xs",
            isHigh ? "text-danger" : isMedium ? "text-amber-400" : "text-primary"
          )}
        >
          {percent}%
        </span>
      </div>

      <Progress value={percent} aria-label="Context used" className="h-1.5" />

      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
        <span>{usedTokens.toLocaleString()} tokens used</span>
        <span>{maxTokens.toLocaleString()} max</span>
      </div>
    </div>
  );
}

export interface ContextContentBodyProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export function ContextContentBody({
  className,
  ...props
}: ContextContentBodyProps) {
  const { usage } = useModelContext();

  const inputTokens = usage?.inputTokens ?? 0;
  const outputTokens = usage?.outputTokens ?? 0;
  const reasoningTokens = usage?.reasoningTokens ?? 0;

  return (
    <div
      className={cn("flex flex-col gap-1.5 py-2 border-y border-white/5", className)}
      {...props}
    >
      <div className="flex justify-between items-center text-xs">
        <span className="text-muted-foreground">Input Tokens</span>
        <span className="font-mono text-foreground font-medium">
          {inputTokens.toLocaleString()}
        </span>
      </div>
      <div className="flex justify-between items-center text-xs">
        <span className="text-muted-foreground">Output Tokens</span>
        <span className="font-mono text-foreground font-medium">
          {outputTokens.toLocaleString()}
        </span>
      </div>
      {reasoningTokens > 0 && (
        <div className="flex justify-between items-center text-[11px] text-muted-foreground/75">
          <span>Reasoning Tokens</span>
          <span className="font-mono text-foreground/80">
            {reasoningTokens.toLocaleString()}
          </span>
        </div>
      )}
    </div>
  );
}

export interface ContextContentFooterProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export function ContextContentFooter({
  className,
  children,
  ...props
}: ContextContentFooterProps) {
  const { modelId, maxTokens } = useModelContext();

  return (
    <div
      className={cn("text-[10px] text-muted-foreground/80 leading-relaxed", className)}
      {...props}
    >
      {children || (
        <span>
          Context window allocated across prompt history and inference tokens on{" "}
          <span className="font-mono text-foreground/90 font-medium">{modelId}</span> ({maxTokens.toLocaleString()} max).
        </span>
      )}
    </div>
  );
}

export interface ContextMeterProps {
  tokens?: number;
  limit?: number;
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  modelId?: string;
  className?: string;
}

export function ContextMeter({
  tokens = 0,
  limit = 20224,
  inputTokens,
  outputTokens,
  reasoningTokens,
  modelId = "qwen3.5-9b",
  className,
}: ContextMeterProps) {
  return (
    <Context
      usedTokens={tokens}
      maxTokens={limit}
      usage={{
        inputTokens,
        outputTokens,
        totalTokens: tokens,
        reasoningTokens,
      }}
      modelId={modelId}
    >
      <ContextTrigger className={className} />
      <ContextContent>
        <ContextContentHeader />
        <ContextContentBody />
        <ContextContentFooter />
      </ContextContent>
    </Context>
  );
}
