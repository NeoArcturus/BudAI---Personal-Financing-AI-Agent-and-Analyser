import React from "react";
import { Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BudAIMessage } from "../BudAIMessage";
import { InterruptHandler } from "./InterruptHandler";
import { EntityHighlighter } from "./EntityHighlighter";

const renderThinkingBadge = (status: string) => {
  return (
    <div className="flex items-center gap-2 bg-black/40 border border-white/5 w-fit px-3 py-1.5 rounded-full text-[10px] text-foreground/50 tracking-widest uppercase transition-all duration-300 animate-pulse">
      <span>{status}</span>
    </div>
  );
};

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
          htil_response: { user_message: string };
        };
      },
    ) => void;
    activeSessionId: string;
    showReasoning: boolean;
  }) => {
    return (
      <div
        className={cn(
          "flex gap-5 w-full",
          message.role === "user" ? "justify-end" : "justify-start",
        )}
      >
        {message.role === "assistant" && (
          <div className="w-8 h-8 rounded-lg bg-content1 border border-white/10 flex items-center justify-center shrink-0">
            <span className="text-foreground/80 font-black text-sm">B</span>
          </div>
        )}

        <div
          className={cn(
            "flex flex-col gap-3 max-w-[85%]",
            message.role === "user" ? "items-end" : "items-start",
          )}
        >
          <div className="flex items-center gap-2 mb-1 px-1">
            <span className="text-[9px] font-black uppercase tracking-widest text-foreground/30">
              {message.role === "user" ? "User" : "BudAI Advisor"}
            </span>
          </div>
          <div
            className={cn(
              "px-6 py-4 rounded-3xl text-[14px] leading-relaxed shadow-sm transition-all",
              message.role === "user"
                ? "bg-content2 backdrop-blur-md border border-white/10 text-foreground font-medium rounded-tr-sm"
                : "bg-content2 backdrop-blur-md border border-white/10 text-foreground/90 font-medium rounded-tl-sm",
            )}
          >
            {message.parts &&
              message.parts.some((p) => p.type.startsWith("data-")) && (
                <div className="flex flex-col gap-2 mb-3">
                  {(() => {
                    const allAnnotations = message.parts
                      .filter((p) => p.type.startsWith("data-"))
                      .flatMap((part) => {
                        const data = (part as { data?: unknown }).data;
                        return Array.isArray(data) ? data : [data];
                      })
                      .filter(
                        Boolean,
                      ) as import("../BudAIMessage").BudAIAnnotation[];

                    const latestThinking = [...allAnnotations]
                      .reverse()
                      .find((ann) => ann.type === "thinking_context");

                    const interrupts = allAnnotations.filter(
                      (ann) => ann.type === "htil_interrupt",
                    );

                    return (
                      <>
                        {latestThinking &&
                          renderThinkingBadge(latestThinking.status)}

                        {interrupts.map((ann, idx) => (
                          <InterruptHandler
                            key={`interrupt-${idx}`}
                            ann={ann}
                            onRespond={(val) => {
                              if (typeof sendMessage === "function") {
                                sendMessage(
                                  {
                                    text: val,
                                  },
                                  {
                                    body: {
                                      session_id:
                                        activeSessionId === "new-session"
                                          ? null
                                          : activeSessionId,
                                      htil_response: {
                                        user_message: val,
                                      },
                                    },
                                  },
                                );
                              }
                            }}
                          />
                        ))}
                      </>
                    );
                  })()}
                </div>
              )}

            {message.parts && message.parts.length > 0
              ? message.parts.map((part, index) => {
                  if (part.type === "text") {
                    const text = part.text;
                    const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/g;
                    const partsArr = [];
                    let lastIndex = 0;
                    let match;
                    while ((match = thinkRegex.exec(text)) !== null) {
                      if (match.index > lastIndex) {
                        partsArr.push({
                          type: "text",
                          content: text.substring(lastIndex, match.index),
                        });
                      }
                      partsArr.push({
                        type: "think",
                        content: match[1],
                      });
                      lastIndex = thinkRegex.lastIndex;
                    }
                    if (lastIndex < text.length) {
                      partsArr.push({
                        type: "text",
                        content: text.substring(lastIndex),
                      });
                    }

                    return (
                      <div key={index}>
                        {partsArr.map((p, i) => {
                          if (p.type === "think") {
                            const isStreamingMessage =
                              status === "streaming" && isLastMessage;
                            if (!showReasoning) {
                              if (
                                isStreamingMessage &&
                                index === message.parts!.length - 1 &&
                                i === partsArr.length - 1
                              ) {
                                return (
                                  <div
                                    key={i}
                                    className="flex items-center gap-3 bg-black/20 border border-white/5 w-fit px-4 py-2 rounded-full mb-3 shadow-[0_0_10px_rgba(255,255,255,0.05)]"
                                  >
                                    <div className="w-2 h-2 rounded-full bg-primary/60 animate-ping"></div>
                                    <span className="text-[9px] text-foreground/50 font-mono tracking-widest uppercase animate-pulse">
                                      Running neuro-stochastic analysis...
                                    </span>
                                  </div>
                                );
                              }
                              return null;
                            }
                            return (
                              <div
                                key={i}
                                className="w-full mb-4 relative flex flex-col group"
                              >
                                <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary/50 via-primary/20 to-transparent rounded-full" />
                                <div className="pl-4 py-1">
                                  <div className="flex items-center gap-2 mb-2 text-white/30 text-[10px] uppercase tracking-widest font-bold">

                                    <span>AI reasoning</span>
                                    {isStreamingMessage &&
                                      index === message.parts!.length - 1 &&
                                      i === partsArr.length - 1 && (
                                        <span className="flex items-center gap-1 text-primary/50 ml-2">
                                          <span className="w-1 h-1 rounded-full bg-primary animate-ping" />
                                          Processing
                                        </span>
                                      )}
                                  </div>
                                  <div className="text-[12px] text-white/50 leading-relaxed max-h-[400px] overflow-y-auto custom-scrollbar font-medium whitespace-pre-wrap">
                                    {p.content}
                                    {isStreamingMessage &&
                                      index === message.parts!.length - 1 &&
                                      i === partsArr.length - 1 && (
                                        <span className="inline-block w-1 h-3 bg-white/30 ml-1 animate-pulse align-middle" />
                                      )}
                                  </div>
                                </div>
                              </div>
                            );
                          }

                          const isStreamingMessage =
                            status === "streaming" && isLastMessage;
                          const isLastPart =
                            index === message.parts!.length - 1 &&
                            i === partsArr.length - 1;
                          const hasActiveInterrupt = message.annotations?.some(
                            (ann) => ann.type === "htil_interrupt",
                          );

                          return (
                            <div
                              key={i}
                              className={cn(
                                "prose prose-invert max-w-none",
                                isStreamingMessage && "animate-hologram-in",
                                isStreamingMessage &&
                                  isLastPart &&
                                  "streaming-cursor",
                                hasActiveInterrupt && "animate-glitch",
                              )}
                            >
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  p: ({ node, children, ...props }) => (
                                    <p {...props}>
                                      <EntityHighlighter>
                                        {children}
                                      </EntityHighlighter>
                                    </p>
                                  ),
                                  li: ({ node, children, ...props }) => (
                                    <li {...props}>
                                      <EntityHighlighter>
                                        {children}
                                      </EntityHighlighter>
                                    </li>
                                  ),
                                  td: ({ node, children, ...props }) => (
                                    <td {...props}>
                                      <EntityHighlighter>
                                        {children}
                                      </EntityHighlighter>
                                    </td>
                                  ),
                                }}
                              >
                                {p.content}
                              </ReactMarkdown>
                            </div>
                          );
                        })}
                      </div>
                    );
                  } else if (part.type === "reasoning") {
                    const isStreamingMessage =
                      status === "streaming" && isLastMessage;
                    if (!showReasoning) {
                      if (
                        isStreamingMessage &&
                        index === message.parts!.length - 1
                      ) {
                        return (
                          <div
                            key={index}
                            className="flex items-center gap-3 bg-black/20 border border-white/5 w-fit px-4 py-2 rounded-full mb-3 shadow-[0_0_10px_rgba(255,255,255,0.05)]"
                          >
                            <div className="w-2 h-2 rounded-full bg-primary/60 animate-ping"></div>
                            <span className="text-[9px] text-foreground/50 font-mono tracking-widest uppercase animate-pulse">
                              Running neuro-stochastic analysis...
                            </span>
                          </div>
                        );
                      }
                      return null;
                    }
                    return (
                      <div
                        key={index}
                        className="w-full mb-4 relative flex flex-col group"
                      >
                        <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary/50 via-primary/20 to-transparent rounded-full" />
                        <div className="pl-4 py-1">
                          <div className="flex items-center gap-2 mb-2 text-white/30 text-[10px] uppercase tracking-widest font-bold">

                            <span>AI reasoning</span>
                            {isStreamingMessage &&
                              index === message.parts!.length - 1 && (
                                <span className="flex items-center gap-1 text-primary/50 ml-2">
                                  <span className="w-1 h-1 rounded-full bg-primary animate-ping" />
                                  Processing
                                </span>
                              )}
                          </div>
                          <div className="text-[12px] text-white/50 leading-relaxed max-h-[400px] overflow-y-auto custom-scrollbar font-medium whitespace-pre-wrap">
                            {part.text}
                            {isStreamingMessage &&
                              index === message.parts!.length - 1 && (
                                <span className="inline-block w-1 h-3 bg-white/30 ml-1 animate-pulse align-middle" />
                              )}
                          </div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                })
              : null}

            {(() => {
              interface TelemetryData {
                type?: string;
                tokens?: number;
                compute_time_ms?: number;
                ttft_ms?: number;
              }

              const allData = [
                ...(message.annotations || []),
                ...(message.parts
                  ?.filter((p) => p.type?.startsWith("data-"))
                  .flatMap((p) => (p as { data?: unknown }).data) || []),
              ].flat(Infinity);

              const telemetry = allData.find(
                (ann) =>
                  ann &&
                  typeof ann === "object" &&
                  (ann as TelemetryData).type === "telemetry",
              ) as TelemetryData | undefined;

              const tokens =
                message.usage?.completionTokens || telemetry?.tokens || 0;

              if (tokens > 0 || telemetry) {
                return (
                  <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-[9px] font-black uppercase tracking-widest text-foreground/40">
                    <div className="flex items-center gap-2">
                      <Zap size={10} className="text-primary/80" />
                      <span>
                        {tokens} TOKENS
                        {telemetry?.compute_time_ms
                          ? ` | ${(telemetry.compute_time_ms / 1000).toFixed(2)}s COMPUTE`
                          : ""}
                        {telemetry?.ttft_ms
                          ? ` | ${telemetry.ttft_ms}ms TTFT`
                          : ""}
                        {telemetry?.compute_time_ms &&
                        telemetry?.ttft_ms &&
                        telemetry.compute_time_ms > telemetry.ttft_ms &&
                        tokens > 0
                          ? ` | ${(tokens / ((telemetry.compute_time_ms - telemetry.ttft_ms) / 1000)).toFixed(1)} T/S`
                          : ""}
                      </span>
                    </div>
                  </div>
                );
              }
              return null;
            })()}
          </div>
        </div>
      </div>
    );
  },
);

MemoizedChatMessage.displayName = "MemoizedChatMessage";
