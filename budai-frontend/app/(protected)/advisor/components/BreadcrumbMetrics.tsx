"use client";

import React, { useState } from "react";
import { Zap, Clock, Copy, Check, ChevronRight, Sliders, Layers } from "lucide-react";
import { Button, Chip } from "@heroui/react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface TelemetryData {
  type?: string;
  tokens?: number;
  compute_time_ms?: number;
  ttft_ms?: number;
  verbosity?: "concise" | "balanced" | "deep_dive";
}

interface BreadcrumbMetricsProps {
  telemetry?: TelemetryData;
  subagents?: string[];
  contentToCopy?: string;
  isStreaming?: boolean;
}

export const BreadcrumbMetrics: React.FC<BreadcrumbMetricsProps> = ({
  telemetry,
  subagents = [],
  contentToCopy,
  isStreaming = false,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!contentToCopy) return;
    navigator.clipboard.writeText(contentToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const tokens = telemetry?.tokens || 0;
  const computeTimeMs = telemetry?.compute_time_ms || 0;
  const ttftMs = telemetry?.ttft_ms || 0;
  const verbosity = telemetry?.verbosity || "balanced";

  const tokensPerSec =
    tokens > 0 && computeTimeMs > ttftMs
      ? (tokens / ((computeTimeMs - ttftMs) / 1000)).toFixed(1)
      : null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="w-full mt-3 pt-3 border-t border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 text-[10px] font-mono select-none"
    >
      {/* Breadcrumb Trail: Preserved custom trail per user directive */}
      <div className="flex items-center flex-wrap gap-1.5 text-foreground/50">
        <span className="flex items-center gap-1 font-semibold text-foreground/70">
          <Layers size={11} className="text-primary/70" />
          <span>BudAI</span>
        </span>

        {subagents.map((agent, i) => (
          <React.Fragment key={i}>
            <ChevronRight size={10} className="text-foreground/30 shrink-0" />
            <span className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/5 text-primary/80 font-mono text-[9px]">
              {agent}
            </span>
          </React.Fragment>
        ))}

        <ChevronRight size={10} className="text-foreground/30 shrink-0" />
        <span className="text-[9px] uppercase tracking-wider text-foreground/40">
          {isStreaming ? "Generating..." : "Synthesized"}
        </span>
      </div>

      {/* Metrics Pills using HeroUI Chip & HeroUI Button for copy */}
      <div className="flex items-center flex-wrap gap-1.5 text-foreground/40 shrink-0">
        {verbosity && (
          <Chip
            size="sm"
            variant="soft"
            className="h-5 px-1.5 bg-white/5 border border-white/5 text-[9px] font-bold uppercase tracking-widest text-foreground/60"
          >
            <span className="flex items-center gap-1">
              <Sliders size={9} />
              <span>{verbosity.replace("_", " ")}</span>
            </span>
          </Chip>
        )}

        {ttftMs > 0 && (
          <Chip
            size="sm"
            variant="soft"
            className="h-5 px-1.5 bg-white/5 border border-white/5 text-[9px] text-foreground/50"
          >
            <span className="flex items-center gap-1">
              <Clock size={9} />
              <span>{ttftMs}ms TTFT</span>
            </span>
          </Chip>
        )}

        {computeTimeMs > 0 && (
          <Chip
            size="sm"
            variant="soft"
            className="h-5 px-1.5 bg-white/5 border border-white/5 text-[9px] text-foreground/50"
          >
            <span>{(computeTimeMs / 1000).toFixed(2)}s</span>
          </Chip>
        )}

        {tokens > 0 && (
          <Chip
            size="sm"
            variant="soft"
            className="h-5 px-1.5 bg-white/5 border border-white/5 text-[9px] text-foreground/50"
          >
            <span className="flex items-center gap-1">
              <Zap size={9} className="text-primary/80" />
              <span>{tokens} tok</span>
            </span>
          </Chip>
        )}

        {tokensPerSec && (
          <Chip
            size="sm"
            variant="soft"
            className="h-5 px-1.5 bg-primary/10 border border-primary/20 text-primary text-[9px] font-bold"
          >
            {tokensPerSec} t/s
          </Chip>
        )}

        {contentToCopy && (
          <Button
            size="sm"
            variant="ghost"
            onPress={handleCopy}
            className="h-5 min-h-5 px-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-[9px] font-mono text-foreground/60 hover:text-foreground transition-all cursor-pointer"
          >
            {copied ? (
              <span className="flex items-center gap-1 text-success">
                <Check size={9} />
                <span className="font-semibold">Copied</span>
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <Copy size={9} />
                <span>Copy</span>
              </span>
            )}
          </Button>
        )}
      </div>
    </motion.div>
  );
};
