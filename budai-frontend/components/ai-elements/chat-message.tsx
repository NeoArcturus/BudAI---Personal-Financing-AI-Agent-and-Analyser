"use client";

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import {
  Message,
  MessageContent,
  Bubble,
  BubbleContent,
  MessageFooter,
} from "./message";
import { Reasoning } from "./reasoning";
import { Task } from "./task";
import { ToolInvocation } from "./tool";
import { Confirmation } from "./confirmation";
import { InlineCitation } from "./inline-citation";
import { DynamicChart } from "./widgets/DynamicChart";
import { TransactionSplitterWidget } from "./widgets/TransactionSplitterWidget";
import { useDashboardStore } from "@/store/dashboardStore";

export interface SubagentStep {
  agent: string;
  action?: string;
  status: "queued" | "running" | "completed" | "failed";
  duration_ms?: number;
}

export interface TelemetryData {
  tokens?: number;
  compute_time_ms?: number;
  ttft_ms?: number;
  verbosity?: string;
}

export interface ChatMessageProps {
  message: {
    id?: string;
    role: "user" | "assistant" | "system" | "data";
    content?: string;
    parts?: any[];
    annotations?: any[];
    toolInvocations?: any[];
  };
  status?: string;
  isLastMessage?: boolean;
  sendMessage?: (
    message: { text: string },
    options?: {
      body: {
        session_id: string | null;
        htil_response?: { user_message: string };
      };
    }
  ) => void;
  activeSessionId?: string;
  showReasoning?: boolean;
}

function sanitizeUserText(text: string): string {
  if (!text) return "";
  return text.replace(/\[STYLE:\s*[^\]]+\]/gi, "").trim();
}

function formatFinancialEntities(children: React.ReactNode): React.ReactNode {
  if (typeof children === "string") {
    const parts = children.split(/(£[0-9.,]+|[0-9.,]+%|\[tx:[^\]]+\])/g);
    return (
      <>
        {parts.map((part, i) => {
          if (part.startsWith("[tx:") && part.endsWith("]")) {
            const txId = part.slice(4, -1).trim();
            return <InlineCitation key={i} txId={txId} />;
          }
          if (part.match(/^(£[0-9.,]+|[0-9.,]+%)$/)) {
            return (
              <span
                key={i}
                className="font-mono font-semibold text-primary inline-block"
              >
                {part}
              </span>
            );
          }
          return part;
        })}
      </>
    );
  }
  if (Array.isArray(children)) {
    return (
      <>
        {children.map((child, i) => (
          <React.Fragment key={i}>
            {formatFinancialEntities(child)}
          </React.Fragment>
        ))}
      </>
    );
  }
  return children;
}

export const ChatMessage = React.memo(
  ({
    message,
    status = "ready",
    isLastMessage = false,
    sendMessage,
    activeSessionId = "",
    showReasoning = false,
  }: ChatMessageProps) => {
    const isAssistant = message.role === "assistant";
    const isStreamingMessage = status === "streaming" && isLastMessage;

    // Aggregate annotations
    const allAnnotations = useMemo(() => {
      const direct = message.annotations || [];
      const fromParts = (message.parts || [])
        .filter((p) => p.type && p.type.startsWith("data-"))
        .flatMap((p) => {
          const d = p.data;
          return Array.isArray(d) ? d : [d];
        })
        .filter(Boolean);
      return [...direct, ...fromParts];
    }, [message.annotations, message.parts]);

    // Subagent step trace
    const subagents: SubagentStep[] = useMemo(() => {
      const stepsMap = new Map<string, SubagentStep>();
      for (const ann of allAnnotations) {
        if (ann.type === "subagent_step") {
          stepsMap.set(ann.agent, {
            agent: ann.agent,
            action: ann.action || stepsMap.get(ann.agent)?.action,
            status: ann.status,
            duration_ms: ann.duration_ms,
          });
        }
      }
      return Array.from(stepsMap.values());
    }, [allAnnotations]);

    // Telemetry
    const telemetry: TelemetryData | null = useMemo(() => {
      for (const ann of allAnnotations) {
        if (ann.type === "telemetry") {
          return ann as TelemetryData;
        }
      }
      return null;
    }, [allAnnotations]);

    // HTIL Interrupts
    const htilInterrupts = useMemo(() => {
      return allAnnotations.filter((ann) => ann.type === "htil_interrupt");
    }, [allAnnotations]);

    // Separate thinking reasoning from final text
    const { cleanText, reasoningText } = useMemo(() => {
      let textAcc = "";
      let reasoningAcc = "";
      const thinkRegex = /<think>([\s\S]*?)<\/think>/g;

      if (message.parts && message.parts.length > 0) {
        for (const part of message.parts) {
          if (part.type === "reasoning" && part.text) {
            reasoningAcc += part.text;
          } else if (part.type === "text" && part.text) {
            let match;
            let lastIndex = 0;
            const raw = part.text;
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

    // User Turn
    if (!isAssistant) {
      const userText = sanitizeUserText(cleanText || message.content || "");
      return (
        <Message align="end">
          <MessageContent>
            <Bubble
              align="end"
              className="bg-primary/15 border border-primary/25 text-foreground rounded-2xl px-4 py-3 text-sm shadow-sm"
            >
              <BubbleContent className="whitespace-pre-wrap leading-relaxed">
                {userText}
              </BubbleContent>
            </Bubble>
          </MessageContent>
        </Message>
      );
    }

    // Assistant Turn
    return (
      <Message align="start" className="gap-3">
        {/* Advisor Avatar */}
        <div className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 border border-primary/30 bg-primary/10 text-primary shadow-[0_0_15px_rgba(0,242,255,0.15)] mt-1">
          <span className="font-mono font-bold text-xs">B</span>
        </div>

        <MessageContent className="space-y-3 max-w-full">
          {/* Header Bar */}
          <div className="flex items-center gap-2 select-none">
            <span className="text-xs font-semibold text-foreground/80">
              BudAI Advisor
            </span>
            {isStreamingMessage && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 border border-primary/25 text-primary">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                Active Analysis
              </span>
            )}
          </div>

          {/* Tool Invocations & Interactive Cards */}
          {message.toolInvocations && message.toolInvocations.length > 0 && (
            <div className="w-full space-y-2">
              {message.toolInvocations.map((tool: any) => {
                const { toolName, toolCallId, args } = tool;

                if (toolName === "generate_ui_chart") {
                  return (
                    <div key={toolCallId} className="w-full my-2">
                      <DynamicChart config={args as any} />
                    </div>
                  );
                }

                if (toolName === "split_transaction") {
                  return (
                    <div key={toolCallId} className="w-full my-2">
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
                                  splits
                                )}`,
                              },
                              { body: { session_id: activeSessionId } as any }
                            );
                          }
                        }}
                        onCancel={() => {
                          if (typeof sendMessage === "function") {
                            sendMessage(
                              {
                                text: `User cancelled split for ${args.transactionId}`,
                              },
                              { body: { session_id: activeSessionId } as any }
                            );
                          }
                        }}
                      />
                    </div>
                  );
                }

                return (
                  <ToolInvocation
                    key={toolCallId}
                    toolName={toolName}
                    state={tool.state || "result"}
                    args={args}
                    result={tool.result}
                  />
                );
              })}
            </div>
          )}

          {/* HTIL Approvals */}
          {htilInterrupts.map((interrupt: any, i: number) => (
            <Confirmation
              key={i}
              title="Authorization Required"
              description={
                interrupt.question ||
                "Authorization required to proceed with this financial operation."
              }
              onApprove={() => {
                if (typeof sendMessage === "function") {
                  sendMessage(
                    { text: "Approved" },
                    {
                      body: {
                        session_id: activeSessionId,
                        htil_response: { user_message: "Approved" },
                      },
                    }
                  );
                }
              }}
              onDeny={() => {
                if (typeof sendMessage === "function") {
                  sendMessage(
                    { text: "Denied" },
                    {
                      body: {
                        session_id: activeSessionId,
                        htil_response: { user_message: "Denied" },
                      },
                    }
                  );
                }
              }}
            />
          ))}

          {/* Chain of Thought Reasoning */}
          {(reasoningText || (isStreamingMessage && !cleanText)) && (
            <Reasoning
              reasoningText={reasoningText}
              isStreaming={isStreamingMessage && !cleanText}
              durationMs={telemetry?.compute_time_ms}
              defaultOpen={showReasoning}
            />
          )}

          {/* Task Pipeline Execution */}
          {subagents.length > 0 && (
            <Task
              steps={subagents.map((s) => ({
                name: s.agent,
                action: s.action,
                status: s.status,
                durationMs: s.duration_ms,
              }))}
            />
          )}

          {/* Advisor Message Response Body */}
          {cleanText ? (
            <div className="w-full rounded-2xl bg-[#0d0e14]/50 border border-white/10 p-5 shadow-lg backdrop-blur-xl">
              <div
                className={cn(
                  "prose prose-invert max-w-none text-sm leading-relaxed font-sans text-foreground/90",
                  isStreamingMessage && "streaming-cursor"
                )}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children, ...props }) => (
                      <p className="mb-3.5 last:mb-0 leading-relaxed" {...props}>
                        {formatFinancialEntities(children)}
                      </p>
                    ),
                    h1: ({ children, ...props }) => (
                      <h1 className="text-base font-bold text-foreground mt-4 mb-2" {...props}>
                        {children}
                      </h1>
                    ),
                    h2: ({ children, ...props }) => (
                      <h2 className="text-sm font-bold text-foreground mt-3 mb-2" {...props}>
                        {children}
                      </h2>
                    ),
                    h3: ({ children, ...props }) => (
                      <h3 className="text-xs font-semibold text-foreground/90 uppercase tracking-wider mt-3 mb-1.5" {...props}>
                        {children}
                      </h3>
                    ),
                    ul: ({ children, ...props }) => (
                      <ul className="space-y-1.5 my-2.5 list-disc list-inside text-foreground/80" {...props}>
                        {children}
                      </ul>
                    ),
                    li: ({ children, ...props }) => (
                      <li className="text-sm leading-relaxed" {...props}>
                        {formatFinancialEntities(children)}
                      </li>
                    ),
                    table: ({ children, ...props }) => (
                      <div className="overflow-x-auto my-3 rounded-xl border border-white/10">
                        <table className="w-full text-xs text-left border-collapse" {...props}>
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children, ...props }) => (
                      <th className="p-2.5 bg-white/5 border-b border-white/10 font-semibold text-foreground" {...props}>
                        {children}
                      </th>
                    ),
                    td: ({ children, ...props }) => (
                      <td className="p-2.5 border-b border-white/5 text-foreground/80" {...props}>
                        {formatFinancialEntities(children)}
                      </td>
                    ),
                    code: ({ children, className, ...props }) => (
                      <code
                        className={cn(
                          "px-1.5 py-0.5 rounded-md bg-white/5 border border-white/10 text-primary font-mono text-xs",
                          className
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
            </div>
          ) : isStreamingMessage && !reasoningText ? (
            <div className="flex items-center gap-2 py-3 px-4 rounded-xl bg-white/[0.02] border border-white/5 text-muted-foreground text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
              <span>Synthesizing portfolio records and spending patterns...</span>
            </div>
          ) : null}

          {/* Action Footer */}
          <MessageFooter
            textToCopy={cleanText}
            onRetry={
              isLastMessage && !isStreamingMessage
                ? () => sendMessage?.({ text: "Please retry the last response" })
                : undefined
            }
          />
        </MessageContent>
      </Message>
    );
  }
);

ChatMessage.displayName = "ChatMessage";
