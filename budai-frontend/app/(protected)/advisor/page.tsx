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
} from "lucide-react";
import {
  Button,
  Avatar,
  Chip,
  ScrollShadow,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { cn } from "@/lib/utils";

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
  } = useBudAI();

  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (sessionIdParam && sessionIdParam !== currentSessionId) {
      setCurrentSessionId(String(sessionIdParam));
    }
  }, [sessionIdParam, currentSessionId, setCurrentSessionId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [sessions, currentSessionId]);

  const activeSession = sessions.find((s) => s.id === currentSessionId);

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isChatLoading) return;

    const msg = input;
    setInput("");
    
    if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
    }

    await sendChatMessage(msg);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
      if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
          textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
      }
  };

  const handleNewChat = () => {
    const id = createNewSession("New Advisory Session");
    router.push(`/advisor?session=${id}`);
  };

  return (
    <div className="flex h-screen w-full bg-[#08090D] font-geist overflow-hidden text-white relative">
      {/* Background Ambience */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden mix-blend-screen opacity-60">
        <div className="absolute -top-[20%] -left-[10%] w-[60%] h-[60%] rounded-full bg-cyan-500/10 blur-[150px]"></div>
        <div className="absolute -bottom-[20%] -right-[10%] w-[60%] h-[60%] rounded-full bg-pink-500/10 blur-[150px]"></div>
      </div>

      {/* Sidebar */}
      <aside className="relative z-10 w-[300px] h-full bg-[#0d1017]/80 backdrop-blur-2xl border-r border-white/5 flex flex-col shrink-0 shadow-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shadow-[inset_0_0_15px_rgba(0,229,255,0.1)]">
                <BrainCircuit size={20} className="text-cyan-400" />
              </div>
              <h2 className="text-white font-bold tracking-tight text-xl">BudAI</h2>
            </div>
            <Button
              isIconOnly
              onPress={() => router.push("/home")}
              variant="ghost"
              className="text-white/40 hover:text-white border-none cursor-pointer"
            >
              <ArrowLeft size={18} />
            </Button>
          </div>

          <Button
            onPress={handleNewChat}
            className="w-full bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl flex items-center justify-start px-4 gap-3 font-medium h-12 mb-6 cursor-pointer transition-all"
          >
            <Plus size={18} className="text-cyan-400" /> New Analysis
          </Button>

          <div className="flex items-center gap-2 text-white/40 px-2 mb-4">
            <History size={14} />
            <span className="text-xs font-bold uppercase tracking-widest">History</span>
          </div>
        </div>

        <ScrollShadow hideScrollBar className="flex-1 px-4 pb-6 space-y-1">
          {sessions.length === 0 ? (
            <div className="text-center py-10 opacity-30">
              <MessageSquare size={32} className="mx-auto mb-3" />
              <p className="text-sm">No recent analysis</p>
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
                  "group relative p-3 rounded-xl cursor-pointer transition-all flex flex-col gap-1 border",
                  currentSessionId === s.id
                    ? "bg-cyan-500/10 border-cyan-500/20 shadow-[0_0_15px_rgba(0,229,255,0.05)]"
                    : "bg-transparent border-transparent hover:bg-white/5",
                )}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={cn(
                      "text-sm font-medium truncate pr-6",
                      currentSessionId === s.id ? "text-cyan-400" : "text-white/80",
                    )}
                  >
                    {String(s.title || "New Chat")}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-white/40 font-mono">
                    <Clock size={10} />
                    <span>
                      {s.lastUpdated
                        ? new Date(String(s.lastUpdated)).toLocaleDateString()
                        : ""}
                    </span>
                  </div>
                <Button
                  isIconOnly
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(String(s.id));
                  }}
                  className="absolute right-2 top-3 opacity-0 group-hover:opacity-100 text-white/30 hover:text-pink-500 hover:bg-pink-500/10 transition-all border-none z-10 w-6 h-6 min-w-6"
                >
                  <Trash2 size={12} />
                </Button>
              </div>
            ))
          )}
        </ScrollShadow>

        <div className="p-6 border-t border-white/5 bg-black/20">
          <div className="flex items-center gap-3">
            <Avatar className="w-10 h-10 bg-linear-to-br from-cyan-500 to-pink-500 shadow-[0_0_15px_rgba(255,51,102,0.2)] text-white font-bold shrink-0 border border-white/10">
              <Avatar.Fallback>
                {String(userName || "U").charAt(0).toUpperCase()}
              </Avatar.Fallback>
            </Avatar>
            <div className="overflow-hidden">
              <p className="text-sm font-semibold truncate text-white">
                {String(userName || "User")}
              </p>
              <p className="text-[10px] text-cyan-400/80 font-mono uppercase tracking-widest">
                Tier: Enterprise
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="relative z-10 flex-1 flex flex-col h-full bg-transparent overflow-hidden">
        
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-8 border-b border-white/5 shrink-0 bg-black/10 backdrop-blur-md">
          <div className="flex flex-col">
            <span className="text-white/80 font-medium tracking-tight text-sm">
              {activeSession?.title ? (activeSession.title as string) : "Financial Orchestration"}
            </span>
            {activeSession?.contextData && (
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                <span className="text-[10px] text-green-500 font-mono uppercase tracking-wider">Live Context</span>
              </div>
            )}
          </div>
          <Chip className="bg-white/5 text-white/40 border border-white/10 px-2 font-mono text-[10px] uppercase tracking-widest">
            V2.0 Engine
          </Chip>
        </header>

        {/* Chat Scroll Area */}
        <ScrollShadow ref={scrollRef} hideScrollBar className="flex-1 w-full relative">
          <div className="max-w-3xl mx-auto w-full px-6 py-10 pb-40">
            {!activeSession ? (
              // Empty State
              <div className="flex flex-col items-center justify-center h-full mt-24 text-center">
                <div className="w-20 h-20 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-[0_0_40px_rgba(0,229,255,0.15)] mb-8">
                  <Sparkles size={32} />
                </div>
                <h1 className="text-3xl font-bold text-white mb-3 tracking-tight">
                  Intelligence Ready
                </h1>
                <p className="text-white/50 max-w-md mx-auto leading-relaxed mb-10">
                  I am connected to your financial graph. Ask me to analyze spending patterns, project future net worth, or audit specific transactions.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-xl">
                  {[
                    "Categorize my recent spending",
                    "Generate a 30-day forecast",
                    "Audit my subscription costs",
                    "Show my health radar",
                  ].map((p, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(p)}
                      className="bg-white/5 border border-white/10 hover:bg-white/10 hover:border-cyan-500/30 text-white/70 hover:text-white rounded-xl p-4 text-sm font-medium transition-all text-left cursor-pointer"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              // Messages
              <div className="space-y-8">
                <AnimatePresence initial={false}>
                  {activeSession.messages.map((m, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      className={cn(
                        "flex gap-5 w-full",
                        m.role === "user" ? "justify-end" : "justify-start"
                      )}
                    >
                      {m.role === "assistant" && (
                        <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(0,229,255,0.1)]">
                          <BrainCircuit size={16} className="text-cyan-400" />
                        </div>
                      )}
                      
                      <div className={cn(
                        "flex flex-col gap-1 max-w-[85%]",
                        m.role === "user" ? "items-end" : "items-start"
                      )}>
                        <div className={cn(
                          "px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed",
                          m.role === "user" 
                            ? "bg-white/10 border border-white/10 text-white rounded-tr-sm" 
                            : "bg-transparent text-white/90"
                        )}>
                          {m.role === "user" ? (
                            <div className="whitespace-pre-wrap">{String(m.text || "")}</div>
                          ) : (
                            <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-[#101115] prose-pre:border prose-pre:border-white/10 prose-headings:text-white prose-a:text-cyan-400 max-w-none">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {String(m.text || "")}
                              </ReactMarkdown>
                            </div>
                          )}
                        </div>
                      </div>

                      {m.role === "user" && (
                         <Avatar className="w-8 h-8 bg-white/10 border border-white/10 text-white font-bold shrink-0">
                          <Avatar.Fallback>
                            {String(userName || "U").charAt(0).toUpperCase()}
                          </Avatar.Fallback>
                        </Avatar>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
                
                {/* Typing Indicator */}
                {isChatLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-5 items-center justify-start w-full"
                  >
                    <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(0,229,255,0.1)]">
                      <Sparkles size={16} className="text-cyan-400" />
                    </div>
                    <div className="flex gap-1.5 px-2 py-3">
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" />
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        </ScrollShadow>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 w-full bg-linear-to-t from-[#08090D] via-[#08090D]/90 to-transparent pt-10 pb-8 px-6">
          <div className="max-w-3xl mx-auto relative">
            <form
              onSubmit={handleSendMessage}
              className="bg-[#101115] border border-white/10 rounded-3xl p-2 pl-4 flex items-end gap-2 shadow-[0_10px_40px_rgba(0,0,0,0.5)] focus-within:border-cyan-500/40 focus-within:shadow-[0_0_20px_rgba(0,229,255,0.1)] transition-all"
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputInput}
                onKeyDown={handleKeyDown}
                placeholder="Message BudAI..."
                rows={1}
                className="flex-1 bg-transparent text-white placeholder:text-white/30 text-[15px] resize-none focus:outline-none focus:ring-0 py-3.5 leading-relaxed max-h-[200px] [&::-webkit-scrollbar]:hidden"
                style={{ overflowY: input.split('\n').length > 5 ? 'auto' : 'hidden' }}
              />
              <Button
                isIconOnly
                type="submit"
                isDisabled={!input.trim() || isChatLoading}
                className={cn(
                  "w-10 h-10 min-w-10 rounded-2xl mb-1.5 transition-all duration-300",
                  input.trim() && !isChatLoading
                    ? "bg-cyan-400 text-black shadow-[0_0_15px_rgba(0,229,255,0.4)] cursor-pointer"
                    : "bg-white/5 text-white/30 cursor-not-allowed",
                )}
              >
                <Send size={16} className={cn(input.trim() && !isChatLoading ? "ml-0.5" : "")} />
              </Button>
            </form>
            <p className="text-center text-[10px] text-white/30 font-mono uppercase tracking-widest mt-4">
              AI CAN MAKE MISTAKES. VERIFY IMPORTANT FINANCIAL DATA.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
