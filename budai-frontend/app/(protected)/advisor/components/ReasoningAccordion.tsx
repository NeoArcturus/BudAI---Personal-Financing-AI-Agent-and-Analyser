"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChevronDown, Brain, Check, Copy, Bot } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button, Chip } from "@heroui/react";
import { cn } from "@/lib/utils";

export interface SubagentStep {
  agent: string;
  action?: string;
  status: "queued" | "running" | "completed" | "failed";
  duration_ms?: number;
}

interface ReasoningAccordionProps {
  reasoningText: string;
  isStreaming: boolean;
  durationMs?: number;
  subagents?: SubagentStep[];
  tokens?: number;
  defaultOpen?: boolean;
}

export const ReasoningAccordion: React.FC<ReasoningAccordionProps> = ({
  reasoningText,
  isStreaming,
  durationMs,
  subagents = [],
  tokens,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen || isStreaming);
  const [copied, setCopied] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(durationMs || 0);
  const startTimeRef = useRef<number | null>(null);

  // Real-time stopwatch while streaming
  useEffect(() => {
    if (isStreaming) {
      if (!startTimeRef.current) {
        startTimeRef.current = Date.now();
      }
      const interval = setInterval(() => {
        if (startTimeRef.current) {
          setElapsedMs(Date.now() - startTimeRef.current);
        }
      }, 100);
      return () => clearInterval(interval);
    } else {
      if (durationMs) {
        setElapsedMs(durationMs);
      }
    }
  }, [isStreaming, durationMs]);

  // Keep accordion open while streaming, collapse on complete if not manually toggled
  useEffect(() => {
    if (isStreaming) {
      setIsOpen(true);
    }
  }, [isStreaming]);

  const handleCopy = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!reasoningText) return;
    navigator.clipboard.writeText(reasoningText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formattedSeconds = (elapsedMs / 1000).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="w-full my-3 border border-white/10 rounded-2xl bg-black/40 backdrop-blur-md overflow-hidden transition-all duration-300 shadow-sm"
    >
      {/* Header bar */}
      <div
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-white/[0.03] transition-colors cursor-pointer select-none"
      >
        <div className="flex items-center gap-3 min-w-0">
          <div
            className={cn(
              "w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-colors",
              isStreaming
                ? "bg-primary/20 text-primary border border-primary/30"
                : "bg-white/5 text-foreground/60 border border-white/10",
            )}
          >
            <Brain
              size={15}
              className={cn(isStreaming && "animate-pulse text-primary")}
            />
          </div>

          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] font-bold uppercase tracking-wider text-foreground/80">
                {isStreaming ? "Reasoning in progress" : "Reasoning process"}
              </span>

              <Chip size="sm" variant="soft" className="h-5 px-1.5 text-[10px] font-mono bg-white/5 border border-white/5 text-foreground/60">
                {formattedSeconds}s{tokens ? ` • ${tokens} tokens` : ""}
              </Chip>

              {isStreaming && (
                <Chip size="sm" variant="soft" className="h-5 px-1.5 text-[9px] font-mono uppercase bg-primary/10 border border-primary/30 text-primary">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping mr-1 inline-block" />
                  Active
                </Chip>
              )}
            </div>

            {/* Subagents inline preview */}
            {subagents.length > 0 && (
              <div className="flex items-center gap-1.5 mt-1.5 overflow-x-auto no-scrollbar">
                <span className="text-[9px] text-foreground/40 uppercase font-mono tracking-tight shrink-0">
                  Subagents:
                </span>
                {subagents.map((sub, idx) => (
                  <Chip
                    key={idx}
                    size="sm"
                    variant="soft"
                    className={cn(
                      "h-5 px-2 text-[9px] font-mono shrink-0 transition-all border",
                      sub.status === "running"
                        ? "bg-primary/15 border-primary/40 text-primary animate-pulse"
                        : sub.status === "completed"
                          ? "bg-white/5 border-white/10 text-foreground/70"
                          : "bg-white/2 border-white/5 text-foreground/40",
                    )}
                  >
                    <span className="flex items-center gap-1">
                      <Bot size={10} />
                      <span>{sub.agent}</span>
                      {sub.status === "completed" && (
                        <Check size={9} className="text-success ml-0.5" />
                      )}
                    </span>
                  </Chip>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 ml-3 shrink-0">
          {reasoningText && (
            <Button
              size="sm"
              variant="ghost"
              isIconOnly
              onPress={() => handleCopy()}
              className="w-7 h-7 min-w-7 rounded-md hover:bg-white/10 text-foreground/40 hover:text-foreground/80 transition-colors"
            >
              {copied ? <Check size={13} className="text-success" /> : <Copy size={13} />}
            </Button>
          )}

          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-foreground/40"
          >
            <ChevronDown size={15} />
          </motion.div>
        </div>
      </div>

      {/* Accordion content */}
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden border-t border-white/5"
          >
            <div className="p-4 bg-black/60">
              {/* Detailed subagent timeline */}
              {subagents.length > 0 && (
                <div className="mb-3 pb-3 border-b border-white/5">
                  <div className="text-[10px] font-mono uppercase tracking-widest text-foreground/40 mb-2">
                    A2A Delegation Trail
                  </div>
                  <div className="space-y-1.5">
                    {subagents.map((step, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between text-[11px] font-mono px-2.5 py-1.5 rounded-lg bg-white/[0.02] border border-white/5"
                      >
                        <div className="flex items-center gap-2">
                          <Bot size={12} className="text-primary/70" />
                          <span className="font-semibold text-foreground/80">
                            {step.agent}
                          </span>
                          {step.action && (
                            <span className="text-foreground/40 text-[10px]">
                              — {step.action}
                            </span>
                          )}
                        </div>
                        <Chip
                          size="sm"
                          variant="soft"
                          className={cn(
                            "h-4 px-1.5 text-[8px] font-mono uppercase tracking-wider",
                            step.status === "running"
                              ? "bg-primary/20 text-primary border border-primary/30 animate-pulse"
                              : step.status === "completed"
                                ? "bg-success/20 text-success border border-success/30"
                                : "text-foreground/30 bg-white/5",
                          )}
                        >
                          {step.status}
                        </Chip>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Reasoning stream text */}
              <div className="relative pl-3 border-l-2 border-primary/30">
                <div className="text-[12px] font-mono text-foreground/70 leading-relaxed max-h-[360px] overflow-y-auto custom-scrollbar whitespace-pre-wrap selection:bg-primary/30">
                  {reasoningText || (
                    <span className="text-foreground/30 italic">
                      Synthesizing mathematical context...
                    </span>
                  )}
                  {isStreaming && (
                    <motion.span
                      animate={{ opacity: [0, 1, 0] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                      className="inline-block w-1.5 h-3.5 bg-primary/80 ml-1 align-middle"
                    />
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
