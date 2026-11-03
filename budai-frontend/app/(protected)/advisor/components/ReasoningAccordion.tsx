"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import { ChevronDown, Brain, Check, Copy, Bot } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button, Chip } from "@heroui/react";
import { cn } from "@/lib/utils";
import { resolveAgentMetadata } from "./AgentHandoverCard";

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
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(durationMs || 0);
  const [isMac, setIsMac] = useState(false);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    setIsMac(/Mac|iPod|iPhone|iPad/.test(navigator.platform || navigator.userAgent));
  }, []);

  // Sync with keyboard toggle
  useEffect(() => {
    setIsOpen(defaultOpen);
  }, [defaultOpen]);

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

  const handleCopy = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!reasoningText) return;
    navigator.clipboard.writeText(reasoningText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formattedSeconds = (elapsedMs / 1000).toFixed(1);

  // Extract a clean thought snippet for the collapsed preview
  const thoughtSnippet = useMemo(() => {
    if (!reasoningText) return "";
    const clean = reasoningText
      .trim()
      .replace(/^Thinking Process:\s*/i, "")
      .replace(/^[#*-\d.\s]+/gm, "")
      .replace(/\n+/g, " ")
      .trim();
    if (clean.length > 95) {
      return `"${clean.substring(0, 92)}..."`;
    }
    return `"${clean}"`;
  }, [reasoningText]);

  return (
    <div className="w-full my-2 rounded-xl border border-white/10 bg-black/30 backdrop-blur-md overflow-hidden transition-all duration-200">
      {/* Header bar */}
      <div
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full px-3.5 py-2.5 flex items-center justify-between text-left hover:bg-white/[0.04] transition-colors cursor-pointer select-none"
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div
            className={cn(
              "w-6 h-6 rounded-lg flex items-center justify-center shrink-0 transition-colors",
              isStreaming
                ? "bg-primary/20 text-primary border border-primary/30"
                : "bg-white/5 text-foreground/60 border border-white/10",
            )}
          >
            <Brain
              size={13}
              className={cn(isStreaming && "animate-pulse text-primary")}
            />
          </div>

          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] font-bold text-foreground/80 tracking-tight">
                {isStreaming ? `Thinking... (${formattedSeconds}s)` : `Thought for ${formattedSeconds}s`}
              </span>

              {tokens ? (
                <Chip size="sm" variant="soft" className="h-4 px-1.5 text-[9px] font-mono bg-white/5 border border-white/5 text-foreground/60">
                  {tokens} tokens
                </Chip>
              ) : null}

              {isStreaming && (
                <span className="flex items-center gap-1 text-[9px] font-mono uppercase text-primary tracking-widest">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                  Live
                </span>
              )}
            </div>

            {/* Collapsed Snippet Preview */}
            {!isOpen && thoughtSnippet && (
              <div className="text-[11px] text-foreground/50 font-sans italic truncate mt-0.5 max-w-xl">
                <span>{thoughtSnippet}</span>
                <span className="ml-1.5 not-italic text-[10px] font-mono text-primary/70 font-semibold">
                  ({isMac ? "⌘O" : "Ctrl+O"} to expand)
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 ml-3 shrink-0">
          {reasoningText && isOpen && (
            <Button
              size="sm"
              variant="ghost"
              isIconOnly
              onPress={() => handleCopy()}
              className="w-6 h-6 min-w-6 rounded-md hover:bg-white/10 text-foreground/40 hover:text-foreground/80 transition-colors"
            >
              {copied ? <Check size={12} className="text-success" /> : <Copy size={12} />}
            </Button>
          )}

          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-foreground/40"
          >
            <ChevronDown size={14} />
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
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-white/5"
          >
            <div className="p-3.5 bg-black/50">
              {/* Specialist Agents timeline */}
              {subagents.length > 0 && (
                <div className="mb-3 pb-2.5 border-b border-white/5">
                  <div className="text-[9px] font-mono uppercase tracking-widest text-foreground/40 mb-2">
                    Specialist Agents Engaged:
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {subagents.map((sub, idx) => {
                      const meta = resolveAgentMetadata(sub.agent);
                      return (
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
                            <span>{meta.name}</span>
                            {sub.status === "completed" && (
                              <Check size={9} className="text-success ml-0.5" />
                            )}
                          </span>
                        </Chip>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Reasoning stream text */}
              <div className="relative pl-3 border-l-2 border-primary/30">
                <div className="text-[12px] font-mono text-foreground/70 leading-relaxed max-h-[360px] overflow-y-auto custom-scrollbar whitespace-pre-wrap selection:bg-primary/30">
                  {reasoningText || (
                    <span className="text-foreground/30 italic">
                      Reviewing financial data...
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
    </div>
  );
};
