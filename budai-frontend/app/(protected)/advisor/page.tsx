"use client";

import React, { useState, useEffect } from "react";

import { Button, Text } from "@heroui/react";
import { useRouter, useSearchParams } from "next/navigation";
import { useChat } from "@ai-sdk/react";
import { BudAIMessage } from "./BudAIMessage";
import { useChatSessions } from "@/lib/hooks";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch, getAuthToken, getApiUrl } from "@/lib/api";
import { DefaultChatTransport } from "ai";

import { AdvisorSidebar, BudAIChatSession } from "./components/AdvisorSidebar";
import { MemoizedChatMessage } from "./components/MemoizedChatMessage";
import { ChatInputArea } from "./components/ChatInputArea";

interface BudAIChatMessage {
  role: "user" | "assistant" | "system" | "data";
  content: string;
  reasoning_content?: string | null;
  timestamp: string;
}

export interface BudAIAdvisorContext {
  type:
    | "cash_flow"
    | "spending_trend"
    | "expense_distribution"
    | "ledger_audit"
    | "market_audit"
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

  useEffect(() => {
    const saved = localStorage.getItem("budai_show_reasoning");
    if (saved === "true") {
      setTimeout(() => setShowReasoning(true), 0);
    }
  }, []);

  const handleToggleReasoning = () => {
    setShowReasoning((prev) => {
      const next = !prev;
      localStorage.setItem("budai_show_reasoning", String(next));
      return next;
    });
  };

  const transport = React.useMemo(() => {
    return new DefaultChatTransport({
      api: getApiUrl("/api/chat/stream"),
      headers: {
        Authorization: `Bearer ${getAuthToken()}`,
      },
    });
  }, []);

  const { messages, setMessages, status, sendMessage, stop, error } =
    useChat<BudAIMessage>({
      id: activeSessionId || "new-session",
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

                // History Sync: Map reasoning_content to a custom reasoning part
                if (msg.reasoning_content) {
                  parts.push({
                    type: "reasoning",
                    text: msg.reasoning_content,
                  });
                }

                // Extract <think> from history just in case it's stored that way
                const text = msg.content || "";
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

                return {
                  id: Math.random().toString(36).substring(7),
                  role: msg.role as "user" | "assistant" | "system" | "data",
                  content: text,
                  parts: parts,
                  createdAt: new Date(msg.timestamp),
                } as BudAIMessage;
              },
            );
            setTimeout(() => {
              setMessages(formattedMessages);
            }, 50);
          }
        } catch (error) {
          console.error("Failed to fetch session messages:", error);
        }
      };

      fetchSessionMessages();
    }
  }, [sessionParam, setMessages]);

  const handleSend = async (text: string) => {
    try {
      if (typeof sendMessage === "function") {
        sendMessage(
          { text: text },
          {
            body: {
              session_id:
                activeSessionId === "new-session" ? null : activeSessionId,
            },
          },
        );
      } else {
        console.error("sendMessage is not a function!");
      }
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await apiFetch(
        "/api/chat/sessions",
        { method: "POST" },
        true,
      );
      const data = (await res.json()) as { session_id?: string };
      if (data && data.session_id) {
        lastFetchedId.current = "";
        setActiveSessionId(data.session_id);
        setMessages([]);
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
        router.push(`/advisor?session=${data.session_id}`);
      }
    } catch (error) {
      console.error("Failed to create new session:", error);
    }
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
    }
  };

  const handleRenameSession = async (id: string, newTitle: string) => {
    try {
      await apiFetch(
        `/api/chat/sessions/${id}`,
        {
          method: "PATCH",
          body: JSON.stringify({ title: newTitle }),
        },
        true,
      );
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
    } catch (error) {
      console.error("Failed to rename session:", error);
    }
  };

  const lastMessage = messages[messages.length - 1];
  const isLocked = Boolean(
    lastMessage?.annotations?.some((ann) => ann.type === "htil_interrupt"),
  );

  return (
    <div className="flex h-screen w-full bg-transparent text-foreground font-sans overflow-hidden transition-colors duration-500">
      <AdvisorSidebar
        sessions={sessions as BudAIChatSession[]}
        sessionsLoading={sessionsLoading}
        activeSessionId={activeSessionId}
        router={router}
        handleNewChat={handleNewChat}
        handleDeleteSession={handleDeleteSession}
        handleRenameSession={handleRenameSession}
      />

      <main className="flex-1 flex flex-col h-full bg-transparent relative overflow-hidden">
        <header className="h-16 flex items-center justify-between py-4 px-6 border-b border-white/10 shrink-0 bg-black/20 backdrop-blur-xl z-20">
          <div className="flex flex-col">
            <div className="flex items-center gap-3 mt-1">

              <Text className="text-foreground font-black text-[13px] uppercase tracking-tight">
                {activeSessionId ? "Advisory Session" : "New Session"}
              </Text>
              {status === "streaming" || status === "submitted" ? (
                <span className="text-[9px] font-black text-primary uppercase tracking-widest opacity-50">
                  Processing...
                </span>
              ) : (
                messages.length > 0 && (
                  <Button
                    size="sm"
                    onPress={() => {
                      setResumeStream(true);
                      setTimeout(() => setResumeStream(false), 500);
                    }}
                    className="bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 h-6 px-3 ml-2 rounded-md text-[9px] uppercase tracking-widest font-bold cursor-pointer transition-all"
                  >
                    Resume Stream
                  </Button>
                )
              )}
            </div>
            {activeSessionId && (
              <span className="text-foreground/30 text-[10px] font-medium tracking-widest uppercase ml-8">
                ID: {activeSessionId.split("-")[0]}
              </span>
            )}
          </div>
          <div
            className="flex items-center gap-3 bg-white/5 border border-white/10 px-3 py-1.5 rounded-full shadow-[0_0_15px_rgba(0,242,255,0.05)] cursor-pointer hover:bg-white/10 transition-colors"
            onClick={handleToggleReasoning}
          >
            <div
              className={`flex items-center h-4 w-7 rounded-full p-0.5 transition-colors ${showReasoning ? "bg-white" : "bg-white/20"}`}
            >
              <div
                className={`h-3 w-3 rounded-full shadow-sm transition-transform ${showReasoning ? "translate-x-3 bg-black" : "translate-x-0 bg-white"}`}
              />
            </div>
            <span className="text-[9px] font-black uppercase tracking-widest text-foreground/50 select-none">
              Show thinking
            </span>
          </div>
        </header>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-4 sm:px-10 py-10 scroll-smooth relative z-10 custom-scrollbar"
        >
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-foreground/30">

              <Text className="text-lg font-bold tracking-tighter uppercase mb-2">
                New Session
              </Text>
              <Text className="text-xs tracking-widest font-medium uppercase opacity-50 text-center max-w-sm leading-relaxed">
                How can I assist with your finances today?
              </Text>
              <div className="grid grid-cols-2 gap-4 w-full max-w-2xl mt-12">
                {[
                  "Analyze my latest spending trends",
                  "What's my cash flow looking like?",
                  "Review my high-value transactions",
                  "Perform a ledger audit",
                ].map((hint, idx) => (
                  <Button
                    key={idx}
                    onPress={() => handleSend(hint)}
                    className="h-16 bg-white/2 border border-white/5 hover:bg-white/5 hover:border-primary/30 flex items-center justify-start px-6 rounded-xl transition-all group cursor-pointer"
                  >
                    <span className="text-[10px] font-black text-foreground/50 group-hover:text-primary uppercase tracking-widest text-left whitespace-normal leading-tight">
                      {hint}
                    </span>
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-8 max-w-4xl mx-auto w-full">
              {messages.map((message, i) => (
                <MemoizedChatMessage
                  key={message.id || i}
                  message={message}
                  status={status}
                  isLastMessage={i === messages.length - 1}
                  sendMessage={sendMessage}
                  activeSessionId={activeSessionId}
                  showReasoning={showReasoning}
                />
              ))}

              {error && (
                <div className="flex justify-center my-4">
                  <div className="bg-danger/20 text-danger text-xs font-black uppercase tracking-widest px-4 py-2 rounded-full border border-danger/30">
                    Connection Error. Retrying...
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="p-4 pb-12 sm:p-10 sm:pb-24 shrink-0 relative z-20">
          <div className="max-w-4xl mx-auto w-full relative">
            <ChatInputArea
              status={status}
              stop={stop}
              onSend={handleSend}
              isLocked={isLocked}
            />
            <div className="text-center mt-4">
              <p className="text-[8px] text-foreground/30 uppercase tracking-[0.3em] font-black">
                AI can make mistakes. Please verify important information.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
