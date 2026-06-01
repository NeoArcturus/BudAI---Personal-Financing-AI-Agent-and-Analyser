"use client";

import React, { useState, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Sparkles,
  History,
  Plus,
  Send,
  Trash2,
  ArrowLeft,
  MessageSquare,
  Clock,
  BrainCircuit,
  ArrowRight,
  Wallet,
} from "lucide-react";
import {
  Button,
  ScrollShadow,
  Surface,
  Input,
  Text,
  Form,
  Skeleton,
  Card,
  ListBox,
} from "@heroui/react";
import { UIMessage } from "@ai-sdk/react";
import { useBudAI } from "@/app/context/AppContext";
import { cn } from "@/lib/utils";
import CoreChartEngine from "@/app/(protected)/_components/CoreChartEngine";
import { NativeChartConfig } from "@/types";

interface ToolInvocation {
  toolName: string;
  toolCallId: string;
  state: string;
  args: Record<string, unknown>;
  result?: { config?: unknown } | unknown;
}

export default function AdvisorPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionIdParam = searchParams.get("session");

  const {
    userName,
    sessions,
    currentSessionId,
    setCurrentSessionId,
    createNewSession,
    deleteSession,
    sendChatMessage,
    isChatLoading,
    chatMessages = [],
    accounts,
  } = useBudAI();

  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sessionIdParam && sessionIdParam !== currentSessionId) {
      const sessionExists = sessions.some(
        (s) => String(s.id) === sessionIdParam,
      );
      if (sessionExists || !currentSessionId) {
        setCurrentSessionId(String(sessionIdParam));
      }
    }
  }, [sessionIdParam, currentSessionId, setCurrentSessionId, sessions]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatMessages, isChatLoading]);

  const activeSession = sessions.find((s) => s.id === currentSessionId);

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isChatLoading) return;

    const msg = input;
    setInput("");

    await sendChatMessage(msg);
  };

  const handleNewChat = () => {
    const id = createNewSession("New Advisory Session");
    router.push(`/advisor?session=${id}`);
  };

  return (
    <Surface
      className="flex h-screen w-full bg-transparent font-sans overflow-hidden text-white relative"
      variant="transparent"
    >
      <Surface
        className="relative z-10 w-64 h-full bg-black/40 backdrop-blur-3xl border-r-[0.5px] border-white/10 flex flex-col shrink-0 shadow-inner obsidian-glass"
        variant="transparent"
      >
        <div className="p-8">
          <div className="flex items-center justify-between mb-12">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center shadow-[0_0_15px_rgba(0,127,255,0.1)]">
                <BrainCircuit size={18} className="text-primary" />
              </div>
              <h1 className="text-foreground font-bold tracking-tighter text-xl uppercase italic m-0">
                BudAI
              </h1>
            </div>
            <Button
              isIconOnly
              onPress={() => router.push("/home")}
              className="text-foreground/20 hover:text-foreground hover:bg-white/5 border-none cursor-pointer rounded-lg w-8 h-8 min-w-8 transition-all flex justify-center items-center"
            >
              <ArrowLeft size={16} />
            </Button>
          </div>

          <Button
            onPress={handleNewChat}
            className="w-full bg-primary/5 border border-primary/20 text-primary hover:bg-primary/10 rounded-xl flex items-center justify-center gap-3 font-black text-[10px] uppercase tracking-widest h-12 mb-10 cursor-pointer transition-all shadow-[0_0_15px_rgba(0,127,255,0.1)] hover:shadow-[0_0_25px_rgba(0,127,255,0.3)]"
          >
            <Plus size={16} /> New Analysis
          </Button>

          <div className="flex items-center gap-3 text-foreground/30 px-2 mb-6">
            <History size={14} />
            <Text className="text-[9px] font-black uppercase tracking-[0.4em]">
              Recent Sessions
            </Text>
          </div>
        </div>

        <ScrollShadow hideScrollBar className="flex-1 px-4 pb-8 space-y-2">
          {sessions.length === 0 ? (
            <div className="text-center py-10 opacity-10">
              <MessageSquare size={32} className="mx-auto mb-4" />
              <Text className="text-[9px] font-black uppercase tracking-widest">
                No Logs Detected
              </Text>
            </div>
          ) : (
            sessions.map((s) => (
              <div
                key={String(s.id)}
                onClick={() => {
                  setCurrentSessionId(String(s.id));
                  router.push(`/advisor?session=${s.id}`);
                }}
                className={cn(
                  "group relative p-4 rounded-xl cursor-pointer transition-all flex flex-col gap-2 border-[0.5px] card-hover",
                  currentSessionId === s.id
                    ? "bg-white/5 border-primary/30 shadow-[0_0_15px_rgba(0,127,255,0.1)]"
                    : "bg-transparent border-transparent hover:border-white/10 hover:bg-white/2",
                )}
              >
                <div className="flex items-center justify-between">
                  <Text
                    className={cn(
                      "text-[11px] font-black uppercase tracking-tight truncate pr-8",
                      currentSessionId === s.id
                        ? "text-primary italic"
                        : "text-foreground/40 group-hover:text-foreground",
                    )}
                  >
                    {String(s.title || "UNDEFINED_SESSION")}
                  </Text>
                </div>
                <div className="flex items-center gap-2 text-[8px] text-foreground/20 font-black uppercase tracking-[0.2em] font-mono">
                  <Clock size={10} />
                  <span>
                    {s.lastUpdated
                      ? new Date(String(s.lastUpdated)).toLocaleDateString(
                          "en-GB",
                          { day: "2-digit", month: "2-digit" },
                        )
                      : "NOW"}
                  </span>
                </div>

                <Button
                  isIconOnly
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(String(s.id));
                  }}
                  className="absolute right-3 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-foreground/20 hover:text-destructive transition-all border-none z-10 w-6 h-6 min-w-6 bg-transparent"
                >
                  <Trash2 size={12} />
                </Button>
              </div>
            ))
          )}
        </ScrollShadow>

        <div className="p-6 border-t-[0.5px] border-white/5 bg-white/1">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-8 h-8 rounded-lg bg-primary/20 border-[0.5px] border-primary/40 flex items-center justify-center text-primary font-black text-[10px]">
                {String(userName || "U")
                  .charAt(0)
                  .toUpperCase()}
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full border border-black shadow-sm" />
            </div>
            <div className="overflow-hidden">
              <Text className="text-[10px] font-black truncate text-foreground uppercase tracking-tight">
                {String(userName || "VERIFIED_USER")}
              </Text>
            </div>
          </div>
        </div>
      </Surface>

      <Surface
        className="relative z-10 flex-1 flex flex-col h-full bg-transparent overflow-hidden"
        variant="transparent"
      >
        <Surface
          className="h-16 flex items-center justify-between px-10 border-b-[0.5px] border-white/5 shrink-0 bg-black/20 backdrop-blur-xl z-20"
          variant="transparent"
        >
          <div className="flex flex-col">
            <Text className="text-foreground/30 font-black text-[9px] uppercase tracking-[0.4em] italic">
              Current Analysis
            </Text>
            <div className="flex items-center gap-3 mt-1">
              <Text className="text-foreground font-black text-[13px] uppercase tracking-tight truncate max-w-64">
                {activeSession?.title
                  ? (activeSession.title as string)
                  : "BudAI Strategic Advisor"}
              </Text>
              <div className="flex items-center gap-2 px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-[8px] font-black uppercase tracking-[0.2em] shadow-[0_0_10px_rgba(0,127,255,0.1)]">
                <span className="w-1 h-1 rounded-full bg-primary animate-pulse" />
                Encrypted Session
              </div>
              {activeSession?.contextData && (
                <div className="flex items-center gap-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/20 shadow-sm">
                  <span className="w-1 h-1 rounded-full bg-green-500 animate-pulse" />
                  <Text className="text-[8px] text-green-500 font-black uppercase tracking-[0.2em]">
                    Live Data
                  </Text>
                </div>
              )}
            </div>
          </div>
        </Surface>

        <ScrollShadow
          ref={scrollRef}
          className="flex-1 w-full relative pt-10 pb-36 overflow-y-auto min-h-0"
        >
          <div className="max-w-4xl mx-auto w-full px-10">
            {!activeSession ? (
              <div className="flex flex-col items-center justify-center h-[70vh] text-center">
                <div className="w-20 h-20 rounded-2xl bg-white/2 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_40px_rgba(0,127,255,0.1)] mb-12 transform rotate-3">
                  <Sparkles size={32} />
                </div>
                <h2 className="text-5xl font-normal text-foreground mb-4 tracking-tighter uppercase italic">
                  Advisor <span className="font-black not-italic">Ready</span>
                </h2>
                <Text className="text-foreground/30 max-w-md mx-auto leading-relaxed mb-16 font-medium tracking-tight text-sm uppercase">
                  Connected to your financial data. How can I help you today?
                </Text>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full max-w-2xl">
                  {[
                    "Simulate MacBook arbitrage (Dubai)",
                    "Audit variable utility spikes",
                    "Project October liquidity floor",
                    "Analyze freelance volatility",
                  ].map((p, i) => (
                    <Button
                      key={i}
                      onPress={() => setInput(p)}
                      className="bg-white/3 border-[0.5px] border-white/5 hover:border-primary/40 text-foreground/40 hover:text-primary rounded-xl p-6 text-[11px] font-black uppercase tracking-widest transition-all text-left cursor-pointer group flex items-center justify-between h-auto shadow-sm"
                    >
                      <span>{p}</span>
                      <ArrowRight
                        size={14}
                        className="opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all"
                      />
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-12">
                <AnimatePresence initial={false}>
                  {chatMessages.map((m) => (
                    <motion.div
                      key={m.id}
                      className={cn(
                        "flex gap-5 w-full animate-fade-in-up",
                        m.role === "user" ? "justify-end" : "justify-start",
                      )}
                    >
                      {m.role === "assistant" && (
                        <div className="w-9 h-9 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center shrink-0 shadow-[0_0_15px_rgba(0,127,255,0.1)] transform -translate-y-1">
                          <BrainCircuit size={18} className="text-primary" />
                        </div>
                      )}

                      <div
                        className={cn(
                          "flex flex-col gap-3 max-w-[85%] relative",
                          m.role === "user" ? "items-end" : "items-start",
                        )}
                      >
                        {m.parts && m.parts.length > 0 ? (
                          m.parts.map((part, index) => {
                            if (part.type === "text") {
                              return (
                                <div
                                  key={`${m.id}-part-${index}`}
                                  className={cn(
                                    "px-8 py-6 rounded-2xl text-[15px] leading-relaxed shadow-inner border-[0.5px] transition-all relative overflow-hidden",
                                    m.role === "user"
                                      ? "bg-primary border-primary/30 text-primary-foreground rounded-tr-sm font-bold tracking-tight"
                                      : "bg-white/3 backdrop-blur-3xl border-white/10 text-foreground/90 rounded-tl-sm shadow-2xl hover:border-primary/40 hover:bg-white/5",
                                  )}
                                >
                                  {m.role === "assistant" && (
                                    <div className="absolute top-0 left-0 right-0 h-px bg-linear-to-r from-primary/40 to-transparent pointer-events-none" />
                                  )}
                                  {m.role === "user" ? (
                                    <p className="whitespace-pre-wrap text-primary-foreground font-bold">
                                      {part.text}
                                    </p>
                                  ) : (
                                    <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/60 prose-pre:border-[0.5px] prose-pre:border-white/10 prose-headings:text-primary prose-a:text-primary max-w-none prose-strong:text-primary/80 prose-code:text-primary/70 text-[14px]">
                                      <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                      >
                                        {part.text}
                                      </ReactMarkdown>
                                    </div>
                                  )}
                                </div>
                              );
                            }

                            if (
                              part.type === "tool-invocation" &&
                              "toolInvocation" in part
                            ) {
                              const ti = part.toolInvocation as ToolInvocation;
                              const { toolName, state, result } = ti;

                              if (toolName === "render_ui_chart") {
                                return (
                                  <motion.div
                                    key={`${m.id}-part-${index}`}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="w-full min-w-150 mt-4"
                                  >
                                    {state !== "output-available" ? (
                                      <Skeleton className="w-full h-80 rounded-3xl bg-white/5" />
                                    ) : (
                                      <div className="p-4 bg-white/2 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
                                        <CoreChartEngine
                                          config={
                                            (
                                              result as {
                                                config: NativeChartConfig;
                                              }
                                            )?.config || null
                                          }
                                        />
                                      </div>
                                    )}
                                  </motion.div>
                                );
                              }

                              if (toolName === "render_account_selector") {
                                return (
                                  <motion.div
                                    key={`${m.id}-part-${index}`}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="mt-4 w-full max-w-sm"
                                  >
                                    <Card className="bg-white/2 backdrop-blur-3xl border border-white/10 rounded-3xl p-6 shadow-2xl">
                                      <div className="flex items-center gap-3 mb-6">
                                        <div className="p-2 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                          <Wallet size={16} />
                                        </div>
                                        <p className="text-xs font-black uppercase tracking-widest text-foreground">
                                          Select Target Accounts
                                        </p>
                                      </div>
                                      <ListBox
                                        selectionMode="multiple"
                                        className="space-y-2"
                                        onSelectionChange={(keys) => {
                                          const selectedIds = Array.from(keys);
                                          if (selectedIds.length > 0) {
                                            const names = selectedIds
                                              .map((id) => {
                                                const acc = accounts.find(
                                                  (a) => a.account_id === id,
                                                );
                                                return (
                                                  acc?.bank_name ||
                                                  acc?.provider_name ||
                                                  id
                                                );
                                              })
                                              .join(", ");
                                            sendChatMessage(
                                              `Analyze these accounts: ${names} (IDs: ${selectedIds.join(", ")})`,
                                            );
                                          }
                                        }}
                                      >
                                        {accounts.map((acc) => (
                                          <ListBox.Item
                                            key={acc.account_id}
                                            id={acc.account_id}
                                            textValue={
                                              acc.bank_name || acc.provider_name
                                            }
                                            className="p-3 rounded-xl hover:bg-white/5 transition-all cursor-pointer border border-transparent hover:border-white/5"
                                          >
                                            <div className="flex flex-col">
                                              <span className="text-[11px] font-bold text-foreground">
                                                {acc.bank_name ||
                                                  acc.provider_name}
                                              </span>
                                              <span className="text-[9px] text-foreground/30 font-mono italic">
                                                {acc.account_number}
                                              </span>
                                            </div>
                                          </ListBox.Item>
                                        ))}
                                      </ListBox>
                                    </Card>
                                  </motion.div>
                                );
                              }
                            }
                            return null;
                          })
                        ) : (
                          <div
                            className={cn(
                              "px-8 py-6 rounded-2xl text-[15px] leading-relaxed shadow-inner border-[0.5px] transition-all relative overflow-hidden",
                              m.role === "user"
                                ? "bg-primary border-primary/30 text-primary-foreground rounded-tr-sm font-bold tracking-tight"
                                : "bg-white/3 backdrop-blur-3xl border-white/10 text-foreground/90 rounded-tl-sm shadow-2xl hover:border-primary/40 hover:bg-white/5",
                            )}
                          >
                            {m.role === "assistant" && (
                              <div className="absolute top-0 left-0 right-0 h-px bg-linear-to-r from-primary/40 to-transparent pointer-events-none" />
                            )}
                            {m.role === "user" ? (
                              <p className="whitespace-pre-wrap text-primary-foreground font-bold">
                                {(m as UIMessage & { content: string })
                                  .content || ""}
                              </p>
                            ) : (
                              <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/60 prose-pre:border-[0.5px] prose-pre:border-white/10 prose-headings:text-primary prose-a:text-primary max-w-none prose-strong:text-primary/80 prose-code:text-primary/70 text-[14px]">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                  {(m as UIMessage & { content: string })
                                    .content || ""}
                                </ReactMarkdown>
                              </div>
                            )}
                          </div>
                        )}
                        <div className="flex items-center gap-4 px-2">
                          <Text className="text-[9px] text-foreground/20 font-black uppercase tracking-[0.3em] font-mono">
                            {(m as { createdAt?: string | Date }).createdAt
                              ? new Date(
                                  (m as { createdAt?: string | Date })
                                    .createdAt!,
                                ).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  hour12: false,
                                })
                              : ""}
                          </Text>
                        </div>
                      </div>

                      {m.role === "user" && (
                        <div className="w-9 h-9 bg-white/5 border-[0.5px] border-white/10 flex items-center justify-center text-foreground font-black text-[10px] shrink-0 rounded-lg transform -translate-y-1">
                          {String(userName || "U")
                            .charAt(0)
                            .toUpperCase()}
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {isChatLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-5 items-start justify-start w-full animate-fade-in-up"
                  >
                    <div className="w-9 h-9 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center shrink-0 shadow-[0_0_15px_rgba(0,127,255,0.1)] transform -translate-y-1">
                      <Sparkles size={18} className="text-primary" />
                    </div>
                    <div className="px-8 py-6 rounded-2xl bg-white/3 backdrop-blur-3xl border border-white/10 text-foreground/40 rounded-tl-sm flex items-center gap-3">
                      <div className="flex gap-1.5">
                        <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse shadow-[0_0_8px_rgba(0,127,255,0.6)]" />
                        <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse shadow-[0_0_8px_rgba(0,127,255,0.6)] [animation-delay:0.2s]" />
                        <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse shadow-[0_0_8px_rgba(0,127,255,0.6)] [animation-delay:0.4s]" />
                      </div>
                      <Text className="text-[11px] font-black uppercase tracking-[0.2em] italic ml-2">
                        Advisor Analysis in Progress...
                      </Text>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </ScrollShadow>

        <Surface
          className="absolute bottom-0 left-0 w-full bg-transparent pt-12 pb-10 px-8 z-20"
          variant="transparent"
        >
          {isChatLoading && (
            <div className="absolute top-0 left-0 w-full h-0.5 overflow-hidden opacity-80 z-30">
              <div className="h-full bg-linear-to-r from-transparent via-primary to-transparent w-1/3 animate-flow-line shadow-[0_0_10px_rgba(0,127,255,0.8)] blur-[1px]"></div>
            </div>
          )}
          <div className="max-w-3xl mx-auto relative">
            <Form
              onSubmit={handleSendMessage}
              className="bg-black/60 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-2xl p-2.5 pl-6 flex items-center gap-4 shadow-2xl focus-within:border-primary/40 transition-all"
            >
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question..."
                className="flex-1 bg-transparent text-foreground placeholder:text-foreground/20 text-[13px] font-black uppercase tracking-widest border-none shadow-none outline-none ring-0 focus:outline-none focus:ring-0"
                variant="secondary"
              />
              <Button
                isIconOnly
                type="submit"
                isDisabled={!input.trim() || isChatLoading}
                className={cn(
                  "w-10 h-10 min-w-10 rounded-xl transition-all duration-300 flex justify-center items-center",
                  input.trim() && !isChatLoading
                    ? "bg-primary text-primary-foreground shadow-lg cursor-pointer hover:scale-105 active:scale-95"
                    : "bg-white/5 text-foreground/20 cursor-not-allowed border-[0.5px] border-white/5",
                )}
              >
                <Send size={16} />
              </Button>
            </Form>
          </div>
        </Surface>
      </Surface>
    </Surface>
  );
}
