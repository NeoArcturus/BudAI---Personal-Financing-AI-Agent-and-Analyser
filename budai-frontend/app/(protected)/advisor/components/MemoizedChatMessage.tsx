"use client";

import React, { useMemo } from "react";
import { Bot, User as UserIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Chip } from "@heroui/react";
import { motion } from "framer-motion";
import { BudAIMessage, BudAIAnnotation, SubagentStepAnnotation, TelemetryAnnotation } from "../BudAIMessage";
import { InterruptHandler } from "./InterruptHandler";
import { EntityHighlighter } from "./EntityHighlighter";
import { DynamicChart } from "./DynamicChart";
import { TransactionSplitterWidget } from "./TransactionSplitterWidget";
import { ReasoningAccordion, SubagentStep } from "./ReasoningAccordion";
import { BreadcrumbMetrics, TelemetryData } from "./BreadcrumbMetrics";
import { AgentHandoverCard } from "./AgentHandoverCard";
import { useDashboardStore } from "@/store/dashboardStore";
import { useEffect } from "react";

function LayoutUpdater({ layout }: { layout: any }) {
  const setLayout = useDashboardStore((state: any) => state.setLayout);
  useEffect(() => {
    setLayout(layout);
  }, [layout, setLayout]);

  return (
    <div className="text-[10px] text-primary/80 font-mono italic my-2 p-2 rounded-lg bg-primary/5 border border-primary/20">
      Dashboard layout reorganized autonomously based on advisory consensus.
    </div>
  );
}

export const MemoizedChatMessage = React.memo(
  ({
    message,
    status,
    isLastMessage,
    sendMessage,
    activeSessionId,
    showReasoning,
  }: {
    message: BudAIMessage;
    status: string;
    isLastMessage: boolean;
    sendMessage: (
      message: { text: string },
      options?: {
        body: {
          session_id: string | null;
          htil_response?: { user_message: string };
        };
      },
    ) => void;
    activeSessionId: string;
    showReasoning: boolean;
  }) => {
    const isAssistant = message.role === "assistant";
    const isStreamingMessage = status === "streaming" && isLastMessage;

    // Collect all annotations across parts and direct message annotations
    const allAnnotations = useMemo(() => {
      const direct = message.annotations || [];
      const fromParts = (message.parts || [])
        .filter((p) => p.type && p.type.startsWith("data-"))
        .flatMap((p) => {
          const d = (p as { data?: unknown }).data;
          return Array.isArray(d) ? d : [d];
        })
        .filter(Boolean);
      return [...direct, ...fromParts] as BudAIAnnotation[];
    }, [message.annotations, message.parts]);

    // Extract subagents and their execution states
    const subagents: SubagentStep[] = useMemo(() => {
      const stepsMap = new Map<string, SubagentStep>();
      for (const ann of allAnnotations) {
        if (ann.type === "subagent_step") {
          const sub = ann as SubagentStepAnnotation;
          stepsMap.set(sub.agent, {
            agent: sub.agent,
            action: sub.action || stepsMap.get(sub.agent)?.action,
            status: sub.status,
            duration_ms: sub.duration_ms,
          });
        }
      }
      return Array.from(stepsMap.values());
    }, [allAnnotations]);

    // Extract telemetry data
    const telemetry: TelemetryData | undefined = useMemo(() => {
      const tel = allAnnotations.find((ann) => ann.type === "telemetry") as
        | TelemetryAnnotation
        | undefined;
      const tokens =
        message.usage?.completionTokens || tel?.tokens || 0;
      if (tokens > 0 || tel) {
        return {
          tokens,
          compute_time_ms: tel?.compute_time_ms,
          ttft_ms: tel?.ttft_ms,
          verbosity: tel?.verbosity,
        };
      }
      return undefined;
    }, [allAnnotations, message.usage]);

    // Extract interrupts
    const interrupts = useMemo(() => {
      return allAnnotations.filter((ann) => ann.type === "htil_interrupt");
    }, [allAnnotations]);

    // Parse text and reasoning content cleanly
    const { cleanText, reasoningText } = useMemo(() => {
      let reasoningAcc = "";
      let textAcc = "";

      const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/g;

      if (message.parts && message.parts.length > 0) {
        for (const part of message.parts) {
          if (part.type === "reasoning") {
            reasoningAcc += (part as { text?: string }).text || "";
          } else if (part.type === "text") {
            const raw = (part as { text?: string }).text || "";
            let match;
            let lastIndex = 0;
            while ((match = thinkRegex.exec(raw)) !== null) {
              if (match.index > lastIndex) {
                textAcc += raw.substring(lastIndex, match.index);
              }
              reasoningAcc += match[1];
              lastIndex = thinkRegex.lastIndex;
            }
            if (lastIndex < raw.length) {
              textAcc += raw.substring(lastIndex);
            }
          }
        }
      } else if (message.content) {
        let match;
        let lastIndex = 0;
        const raw = message.content;
        while ((match = thinkRegex.exec(raw)) !== null) {
          if (match.index > lastIndex) {
            textAcc += raw.substring(lastIndex, match.index);
          }
          reasoningAcc += match[1];
          lastIndex = thinkRegex.lastIndex;
        }
        if (lastIndex < raw.length) {
          textAcc += raw.substring(lastIndex);
        }
      }

      return {
        cleanText: textAcc.trim(),
        reasoningText: reasoningAcc.trim(),
      };
    }, [message.parts, message.content]);

    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className={cn(
          "w-full flex transition-all duration-300",
          isAssistant ? "justify-start" : "justify-end",
        )}
      >
        <div
          className={cn(
            "flex gap-4 max-w-4xl w-full",
            isAssistant ? "flex-row items-start" : "flex-row-reverse items-start",
          )}
        >
          {/* Avatar Icon */}
          <div
            className={cn(
              "w-8 h-8 rounded-xl flex items-center justify-center shrink-0 border transition-all mt-1",
              isAssistant
                ? "bg-primary/10 text-primary border-primary/30 shadow-[0_0_15px_rgba(0,242,255,0.15)]"
                : "bg-white/10 text-foreground/80 border-white/15",
            )}
          >
            {isAssistant ? (
              <span className="font-mono font-black text-xs">B</span>
            ) : (
              <UserIcon size={14} />
            )}
          </div>

          {/* Message Card Body */}
          <div
            className={cn(
              "flex flex-col min-w-0 flex-1",
              isAssistant ? "items-start" : "items-end",
            )}
          >
            {/* Header / Role Bar */}
            <div className="flex items-center gap-2 mb-1.5 px-1 select-none">
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-foreground/40">
                {isAssistant ? "BudAI Autonomous Advisor" : "User"}
              </span>
              {isStreamingMessage && (
                <Chip size="sm" variant="soft" className="h-4 px-1.5 bg-primary/10 border border-primary/30 text-primary text-[8px] font-mono uppercase tracking-widest">
                  <span className="w-1 h-1 rounded-full bg-primary animate-ping mr-1 inline-block" />
                  Generating
                </Chip>
              )}
            </div>

            {/* Free-form Container for AI, Sleek Bubble for User */}
            <div
              className={cn(
                "w-full transition-all duration-200",
                isAssistant
                  ? "bg-transparent border-0 shadow-none px-0 py-1 text-foreground"
                  : "rounded-2xl px-5 py-3.5 bg-content2/80 backdrop-blur-md border border-white/15 text-foreground max-w-2xl shadow-md",
              )}
            >
              {/* Tool & Subagent Invocations for Assistant */}
              {isAssistant && message.toolInvocations && message.toolInvocations.length > 0 && (
                <div className="w-full my-2 space-y-2">
                  {message.toolInvocations.map((tool: any) => {
                    const { toolName, toolCallId, args } = tool;

                    if (toolName === "generate_ui_chart") {
                      return (
                        <div key={toolCallId} className="w-full my-3">
                          <DynamicChart config={args as any} />
                        </div>
                      );
                    }

                    if (toolName === "reorder_dashboard_widgets") {
                      return <LayoutUpdater key={toolCallId} layout={args.layout} />;
                    }

                    if (toolName === "split_transaction") {
                      return (
                        <div key={toolCallId} className="w-full my-3">
                          <TransactionSplitterWidget
                            transactionId={args.transactionId}
                            originalAmount={args.originalAmount}
                            merchantName={args.merchantName}
                            suggestedSplits={args.suggestedSplits}
                            onConfirm={(splits) => {
                              if (typeof sendMessage === "function") {
                                sendMessage(
                                  {
                                    text: `User confirmed split for ${args.transactionId} with ${JSON.stringify(
                                      splits,
                                    )}`,
                                  },
                                  { body: { session_id: activeSessionId } as any },
                                );
                              }
                            }}
                            onCancel={() => {
                              if (typeof sendMessage === "function") {
                                sendMessage(
                                  {
                                    text: `User cancelled split for ${args.transactionId}`,
                                  },
                                  { body: { session_id: activeSessionId } as any },
                                );
                              }
                            }}
                          />
                        </div>
                      );
                    }

                    // For all other specialist agents / data retrieval tools: Render Handover Card
                    return (
                      <AgentHandoverCard
                        key={toolCallId}
                        toolName={toolName}
                        args={args}
                        state={tool.state}
                        result={tool.result}
                      />
                    );
                  })}
                </div>
              )}

              {/* Interrupt Handler (Human in the loop) */}
              {interrupts.map((ann, idx) => (
                <InterruptHandler
                  key={`interrupt-${idx}`}
                  ann={ann}
                  onRespond={(val) => {
                    if (typeof sendMessage === "function") {
                      sendMessage(
                        { text: val },
                        {
                          body: {
                            session_id: activeSessionId,
                            htil_response: { user_message: val },
                          },
                        },
                      );
                    }
                  }}
                />
              ))}

              {/* Reasoning Accordion (Deep Thinking & A2A Subagents) */}
              {isAssistant && (reasoningText || (isStreamingMessage && subagents.length > 0)) && (
                <ReasoningAccordion
                  reasoningText={reasoningText}
                  isStreaming={isStreamingMessage}
                  durationMs={telemetry?.compute_time_ms}
                  subagents={subagents}
                  tokens={telemetry?.tokens}
                  defaultOpen={showReasoning}
                />
              )}

              {/* Main Text Content */}
              {cleanText ? (
                <div
                  className={cn(
                    "prose prose-invert max-w-none text-[13.5px] leading-relaxed font-sans",
                    isStreamingMessage && "streaming-cursor",
                  )}
                >
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children, ...props }) => (
                        <p className="mb-3 last:mb-0" {...props}>
                          <EntityHighlighter>{children}</EntityHighlighter>
                        </p>
                      ),
                      li: ({ children, ...props }) => (
                        <li className="my-1" {...props}>
                          <EntityHighlighter>{children}</EntityHighlighter>
                        </li>
                      ),
                      td: ({ children, ...props }) => (
                        <td className="p-2 border border-white/10" {...props}>
                          <EntityHighlighter>{children}</EntityHighlighter>
                        </td>
                      ),
                      th: ({ children, ...props }) => (
                        <th className="p-2 border border-white/10 font-bold bg-white/5" {...props}>
                          {children}
                        </th>
                      ),
                      code: ({ children, className, ...props }) => (
                        <code
                          className={cn(
                            "px-1.5 py-0.5 rounded bg-black/40 border border-white/10 text-primary font-mono text-[12px]",
                            className,
                          )}
                          {...props}
                        >
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {cleanText}
                  </ReactMarkdown>
                </div>
              ) : isStreamingMessage && !reasoningText ? (
                <div className="flex items-center gap-2 py-2 text-foreground/40 font-mono text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                  <span>Reviewing transactions and accounts...</span>
                </div>
              ) : null}

              {/* Breadcrumb Metrics & Telemetry Footer */}
              {isAssistant && (cleanText || telemetry) && (
                <BreadcrumbMetrics
                  telemetry={telemetry}
                  subagents={subagents.map((s) => s.agent)}
                  contentToCopy={cleanText}
                  isStreaming={isStreamingMessage}
                />
              )}
            </div>
          </div>
        </div>
      </motion.div>
    );
  },
);

MemoizedChatMessage.displayName = "MemoizedChatMessage";
