"use client";

import React, { useState, useEffect } from "react";

import { Button, Text, toast } from "@heroui/react";
import { useRouter, useSearchParams } from "next/navigation";
import { useChat } from "@ai-sdk/react";
import { BudAIMessage } from "./BudAIMessage";
import { useChatSessions } from "@/lib/hooks";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch, getAuthToken, getApiUrl } from "@/lib/api";
import { DefaultChatTransport } from "ai";

import {
  MessageScrollerProvider,
  MessageScroller,
  MessageScrollerViewport,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerButton,
  ContextMeter,
  PromptInput,
  DetailLevel,
  ChatMessage,
  SessionSidebar,
} from "@/components/ai-elements";
import { TrendingUp, LineChart, ShieldCheck, Wallet, RotateCw } from "lucide-react";

export type VerbosityMode = DetailLevel;

interface BudAIChatMessage {
  role: "user" | "assistant" | "system" | "data";
  content: string;
  reasoning_content?: string | null;
  timestamp: string;
  ttft_ms?: number | null;
  compute_time_ms?: number | null;
  tokens?: number | null;
}

export interface BudAIAdvisorContext {
  type:
  | "cash_flow"
  | "spending_trend"
  | "expense_distribution"
  | "transactions"
  | "market_insights"
  | "general";
  accountId?: string;
  data?: [];
}

export default function AdvisorPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const sessionParam = searchParams.get("session");

  const { data: sessions = [], isLoading: sessionsLoading } = useChatSessions();

  const [activeSessionId, setActiveSessionId] = useState(sessionParam || "");

  const lastFetchedId = React.useRef("");

  const [resumeStream, setResumeStream] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [verbosity, setVerbosity] = useState<VerbosityMode>("balanced");

  useEffect(() => {
    const saved = localStorage.getItem("budai_show_reasoning");
    if (saved === "true") {
      setTimeout(() => setShowReasoning(true), 0);
    }
    const savedVerbosity = localStorage.getItem("budai_chat_verbosity") as VerbosityMode;
    if (savedVerbosity && ["concise", "balanced", "deep_dive"].includes(savedVerbosity)) {
      setVerbosity(savedVerbosity);
    }
  }, []);

  const handleVerbosityChange = (mode: VerbosityMode) => {
    setVerbosity(mode);
    localStorage.setItem("budai_chat_verbosity", mode);
  };

  const handleToggleReasoning = () => {
    setShowReasoning((prev) => {
      const next = !prev;
      localStorage.setItem("budai_show_reasoning", String(next));
      return next;
    });
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac =
        typeof window !== "undefined" &&
        /Mac|iPod|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
      const isModifierActive = isMac
        ? e.metaKey && !e.ctrlKey
        : e.ctrlKey && !e.metaKey;

      if (isModifierActive && e.key.toLowerCase() === "o") {
        e.preventDefault();
        handleToggleReasoning();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const transport = React.useMemo(() => {
    return new DefaultChatTransport({
      api: getApiUrl("/api/chat/stream"),
      headers: () => ({
        Authorization: `Bearer ${getAuthToken()}`,
      }),
    });
  }, []);

  const { messages, setMessages, status, sendMessage, stop, error } =
    useChat<BudAIMessage>({
      id: "budai-advisor-chat",
      transport,
      onError: (err) => {
        if (err.message.includes("401")) {
          window.dispatchEvent(new Event("budai-unauthorized"));
        }
      },
      resume: resumeStream,
    });

  const scrollRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (sessionParam && sessionParam !== lastFetchedId.current) {
      // Guard against interrupting an active streaming response
      if (status === "streaming" || status === "submitted") {
        return;
      }

      lastFetchedId.current = sessionParam;

      const fetchSessionMessages = async () => {
        try {
          const res = await apiFetch(
            `/api/chat/sessions/${sessionParam}`,
            {},
            true,
          );
          const data = (await res.json()) as { messages?: unknown[] };
          if (
            data &&
            typeof data === "object" &&
            Array.isArray(data.messages)
          ) {
            setActiveSessionId(sessionParam);
            const formattedMessages: BudAIMessage[] = data.messages.map(
              (m: unknown) => {
                const msg = m as BudAIChatMessage;
                const parts: Array<{ type: "text" | "reasoning"; text: string }> = [];

                if (msg.reasoning_content) {
                  parts.push({
                    type: "reasoning",
                    text: msg.reasoning_content,
                  });
                }

                let text = msg.content || "";
                if (msg.role === "user") {
                  text = text.replace(/\[STYLE:\s*[^\]]+\]/gi, "").trim();
                }
                const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/g;
                let hasThinkTags = false;
                let match;
                let lastIndex = 0;

                while ((match = thinkRegex.exec(text)) !== null) {
                  hasThinkTags = true;
                  if (match.index > lastIndex) {
                    parts.push({
                      type: "text",
                      text: text.substring(lastIndex, match.index),
                    });
                  }
                  parts.push({ type: "reasoning", text: match[1] });
                  lastIndex = thinkRegex.lastIndex;
                }

                if (hasThinkTags) {
                  if (lastIndex < text.length) {
                    parts.push({
                      type: "text",
                      text: text.substring(lastIndex),
                    });
                  }
                } else {
                  parts.push({ type: "text", text: text });
                }

                const annotations = [];
                if (msg.ttft_ms || msg.compute_time_ms || msg.tokens) {
                  annotations.push({
                    type: "telemetry",
                    ttft_ms: msg.ttft_ms,
                    compute_time_ms: msg.compute_time_ms,
                    tokens: msg.tokens,
                  });
                }

                return {
                  id: Math.random().toString(36).substring(7),
                  role: msg.role as "user" | "assistant" | "system" | "data",
                  content: text,
                  parts: parts,
                  annotations: annotations,
                  createdAt: new Date(msg.timestamp),
                } as BudAIMessage;
              },
            );
            setMessages(formattedMessages);
          }
        } catch (error) {
          console.error("Failed to fetch session messages:", error);
          toast.danger("Failed to fetch chat history");
        }
      };

      fetchSessionMessages();
    }
  }, [sessionParam, status, setMessages]);

  const handleSend = async (text: string, selectedVerbosity: VerbosityMode = verbosity) => {
    try {
      if (typeof sendMessage === "function") {
        let currentSessionId = activeSessionId;

        if (!currentSessionId || currentSessionId === "new-session") {
          currentSessionId = crypto.randomUUID();
          setActiveSessionId(currentSessionId);
          lastFetchedId.current = currentSessionId;
          if (typeof window !== "undefined") {
            window.history.replaceState(null, "", `/advisor?session=${currentSessionId}`);
          }
        }

        sendMessage(
          { text: text },
          {
            body: {
              session_id: currentSessionId,
              verbosity: selectedVerbosity,
            },
          },
        );
      } else {
        console.error("sendMessage is not a function!");
      }
    } catch (error) {
      console.error("Failed to send message:", error);
      toast.danger("Failed to send message");
    }
  };

  const promptParam = searchParams.get("prompt");
  const initialPromptHandled = React.useRef(false);

  useEffect(() => {
    if (promptParam && !initialPromptHandled.current) {
      initialPromptHandled.current = true;
      if (typeof window !== "undefined") {
        const cleanUrl = sessionParam ? `/advisor?session=${sessionParam}` : `/advisor`;
        window.history.replaceState(null, "", cleanUrl);
      }
      handleSend(promptParam);
    }
  }, [promptParam, sessionParam]);

  const handleNewChat = () => {
    lastFetchedId.current = "";
    setActiveSessionId("");
    setMessages([]);
    router.push("/advisor");
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await apiFetch(`/api/chat/sessions/${id}`, { method: "DELETE" }, true);
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      if (activeSessionId === id) {
        setActiveSessionId("");
        setMessages([]);
        router.push("/advisor");
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      toast.danger("Failed to delete chat");
    }
  };

  const handleRenameSession = async (id: string, newTitle: string) => {
    try {
      await apiFetch(
        `/api/chat/sessions/${id}`,
        {
          method: "PUT",
          body: JSON.stringify({ title: newTitle }),
        },
        true,
      );
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
    } catch (error) {
      console.error("Failed to rename session:", error);
      toast.danger("Failed to rename chat");
    }
  };

  const lastMessage = messages[messages.length - 1];
  const isLocked = Boolean(
    lastMessage?.annotations?.some((ann) => ann.type === "htil_interrupt"),
  );

  const { currentTokens, currentLimit, inputTokens, outputTokens, reasoningTokens } = React.useMemo(() => {
    let tokens = 0;
    let limit = 20224;
    let inTokens = 0;
    let outTokens = 0;
    let rTokens = 0;

    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];

      // Check annotations
      const tel = msg.annotations?.find((a) => a.type === "telemetry") as
        | { tokens?: number; context_limit?: number; input_tokens?: number; output_tokens?: number; reasoning_tokens?: number }
        | undefined;
      if (tel?.tokens) {
        tokens = tel.tokens;
        if (tel.context_limit) limit = tel.context_limit;
        if (tel.input_tokens) inTokens = tel.input_tokens;
        if (tel.output_tokens) outTokens = tel.output_tokens;
        if (tel.reasoning_tokens) rTokens = tel.reasoning_tokens;
        break;
      }

      // Check stream parts
      if (msg.parts && Array.isArray(msg.parts)) {
        for (const p of msg.parts) {
          const d = (p as any).data;
          const list = Array.isArray(d) ? d : [d];
          for (const item of list) {
            if (item && item.type === "telemetry") {
              if (item.tokens) tokens = item.tokens;
              if (item.context_limit) limit = item.context_limit;
              if (item.input_tokens) inTokens = item.input_tokens;
              if (item.output_tokens) outTokens = item.output_tokens;
              if (item.reasoning_tokens) rTokens = item.reasoning_tokens;
              break;
            }
          }
          if (tokens > 0) break;
        }
      }
      if (tokens > 0) break;

      // Check message metadata usage from Vercel AI SDK
      const metaUsage = (msg as any).metadata?.usage;
      if (metaUsage) {
        inTokens = metaUsage.inputTokens ?? metaUsage.promptTokens ?? 0;
        outTokens = metaUsage.outputTokens ?? metaUsage.completionTokens ?? 0;
        tokens = metaUsage.totalTokens ?? (inTokens + outTokens);
        break;
      }

      // Check usage fallback
      if (msg.usage?.totalTokens) {
        tokens = msg.usage.totalTokens;
        outTokens = msg.usage.completionTokens ?? 0;
        inTokens = msg.usage.promptTokens ?? Math.max(0, tokens - outTokens);
        break;
      }
      if (msg.usage?.completionTokens) {
        outTokens = msg.usage.completionTokens;
        tokens = outTokens;
        break;
      }
    }

    if (tokens > 0 && inTokens === 0 && outTokens === 0) {
      outTokens = tokens;
    }

    return {
      currentTokens: tokens,
      currentLimit: limit,
      inputTokens: inTokens,
      outputTokens: outTokens,
      reasoningTokens: rTokens,
    };
  }, [messages]);

  return (
    <div className="flex h-full w-full bg-background text-foreground font-sans overflow-hidden transition-colors duration-500">
      <SessionSidebar
        sessions={sessions as any}
        sessionsLoading={sessionsLoading}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          router.push(`/advisor?session=${id}`);
        }}
        onNewSession={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
      />

      <main className="flex-1 flex flex-col h-full bg-gradient-to-b from-[#07080c] via-[#090b11] to-[#040508] relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_80%_60%_at_50%_-20%,rgba(0,242,255,0.06),transparent)]" />

        <header className="h-16 flex items-center justify-between py-4 px-6 border-b border-white/10 shrink-0 bg-[#0d0e14]/60 backdrop-blur-xl z-20">
          <div className="flex flex-col">
            <div className="flex items-center gap-3">
              <span className="text-foreground font-bold text-sm tracking-tight">
                {activeSessionId ? "Advisory Session" : "Financial Consultation"}
              </span>
              {status === "streaming" || status === "submitted" ? (
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 border border-primary/25 text-primary">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                  Analyzing
                </span>
              ) : (
                messages.length > 0 && (
                  <button
                    type="button"
                    onClick={() => {
                      setResumeStream(true);
                      setTimeout(() => setResumeStream(false), 500);
                    }}
                    className="bg-primary/10 text-primary hover:bg-primary/20 border border-primary/25 h-6 px-2.5 rounded-md text-[10px] font-medium cursor-pointer transition-all"
                  >
                    Resume Stream
                  </button>
                )
              )}
            </div>
            {activeSessionId && (
              <span className="text-muted-foreground/60 text-[10px] font-mono mt-0.5">
                Session: {activeSessionId.split("-")[0]}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <ContextMeter
              tokens={currentTokens}
              limit={currentLimit}
              inputTokens={inputTokens}
              outputTokens={outputTokens}
              reasoningTokens={reasoningTokens}
            />
          </div>
        </header>

        <MessageScrollerProvider autoScroll>
          <MessageScroller>
            <MessageScrollerViewport>
              <MessageScrollerContent>
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 px-4 max-w-3xl mx-auto w-full text-center">
                    <div className="w-12 h-12 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 text-primary shadow-[0_0_25px_rgba(0,242,255,0.15)]">
                      <TrendingUp size={22} />
                    </div>
                    <h2 className="text-lg font-bold text-foreground mb-1 tracking-tight">
                      BudAI Advisory Intelligence
                    </h2>
                    <p className="text-xs text-muted-foreground max-w-md leading-relaxed mb-8">
                      Autonomous financial consultation, risk evaluation, and multi-account portfolio analysis.
                    </p>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
                      {[
                        {
                          title: "Analyze Spending Patterns",
                          desc: "Identify monthly trends, anomalies, and recurring rhythms.",
                          icon: <TrendingUp size={16} className="text-primary shrink-0" />,
                          prompt: "Analyze my latest spending trends and highlight any anomalies.",
                        },
                        {
                          title: "Cash Flow & Runway",
                          desc: "Evaluate liquidity trajectory and upcoming obligations.",
                          icon: <LineChart size={16} className="text-blue-400 shrink-0" />,
                          prompt: "What is my cash flow looking like and how is my liquidity runway?",
                        },
                        {
                          title: "Audit High-Value Outflows",
                          desc: "Review significant single purchases and transfers.",
                          icon: <ShieldCheck size={16} className="text-emerald-400 shrink-0" />,
                          prompt: "Review my highest-value transactions and check for unexpected spikes.",
                        },
                        {
                          title: "Portfolio Overview",
                          desc: "Consolidated breakdown across connected bank accounts.",
                          icon: <Wallet size={16} className="text-purple-400 shrink-0" />,
                          prompt: "Give me an account summary across all my connected accounts.",
                        },
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          onClick={() => handleSend(item.prompt)}
                          className="group p-4 rounded-2xl bg-[#0d0e14]/50 hover:bg-white/[0.04] border border-white/10 hover:border-primary/40 transition-all cursor-pointer shadow-sm flex items-start gap-3.5 backdrop-blur-md"
                        >
                          <div className="p-2 rounded-xl bg-white/[0.03] border border-white/5 group-hover:bg-primary/10 group-hover:border-primary/20 transition-all">
                            {item.icon}
                          </div>
                          <div className="min-w-0 flex-1">
                            <h3 className="text-xs font-bold text-foreground group-hover:text-primary transition-colors">
                              {item.title}
                            </h3>
                            <p className="text-[11px] text-muted-foreground mt-0.5 leading-snug">
                              {item.desc}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex items-start justify-between pb-5 border-b border-white/5 mb-6">
                      <div>
                        <h1 className="text-base font-semibold text-foreground tracking-tight">
                          {activeSessionId ? "Advisory Thread" : "New Thread"}
                        </h1>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Inquire about allocations, risk exposures, or cash flow.
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={handleNewChat}
                        className="w-8 h-8 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 flex items-center justify-center text-muted-foreground hover:text-foreground transition-all cursor-pointer"
                        title="New Thread"
                      >
                        <RotateCw size={14} />
                      </button>
                    </div>

                    {messages.map((message, i) => (
                      <MessageScrollerItem
                        key={message.id || i}
                        messageId={message.id}
                        scrollAnchor={message.role === "user"}
                      >
                        <ChatMessage
                          message={message}
                          status={status}
                          isLastMessage={i === messages.length - 1}
                          sendMessage={sendMessage}
                          activeSessionId={activeSessionId}
                          showReasoning={showReasoning}
                        />
                      </MessageScrollerItem>
                    ))}

                    {error && (
                      <div className="flex justify-center my-4">
                        <div className="bg-danger/15 text-danger text-xs font-semibold px-4 py-2 rounded-full border border-danger/30">
                          Connection temporarily interrupted. Retrying...
                        </div>
                      </div>
                    )}
                  </>
                )}
              </MessageScrollerContent>
            </MessageScrollerViewport>
            <MessageScrollerButton />
          </MessageScroller>
        </MessageScrollerProvider>

        <div className="px-4 pt-2 pb-28 sm:px-8 sm:pb-32 shrink-0 relative z-20">
          <div className="max-w-4xl mx-auto w-full relative">
            <PromptInput
              onSend={handleSend}
              isGenerating={status === "streaming" || status === "submitted"}
              onStop={stop}
              isLocked={isLocked}
              detailLevel={verbosity}
              onDetailLevelChange={handleVerbosityChange}
            />
            <div className="text-center mt-2.5">
              <p className="text-[10px] text-muted-foreground/50 font-normal">
                BudAI provides automated data insights. Verify critical transactions before executing external actions.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
