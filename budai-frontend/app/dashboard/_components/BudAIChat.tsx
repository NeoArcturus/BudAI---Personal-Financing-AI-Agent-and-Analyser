// _components/BudAIChat.tsx
"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Loader2,
  Sparkles,
  TrendingUp,
  ShieldCheck,
  Activity,
  CalendarDays,
  BarChart2,
  Flame,
} from "lucide-react";
import { ChatMessage, Account, TabType } from "@/types";
import ReactMarkdown from "react-markdown";

interface BudAIChatProps {
  onAiAction: (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
    extraParam?: string,
  ) => void;
  activeAccountId: string | null;
  accounts: Account[];
}

export default function BudAIChat({
  onAiAction,
  activeAccountId,
  accounts,
}: BudAIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  const handleSend = async (
    overrideText?: string,
    actionType?: TabType | string,
  ) => {
    const textToSend = overrideText || input;
    if (!textToSend.trim() || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: textToSend }]);
    setInput("");
    setLoading(true);

    const activeAccount = accounts.find(
      (a) =>
        a.account_id === activeAccountId ||
        a.truelayer_account_id === activeAccountId,
    );

    const bankName =
      activeAccountId === "ALL"
        ? "ALL"
        : activeAccount
          ? activeAccount.bank_name ||
            activeAccount.provider_name ||
            "Unknown Bank"
          : "Unknown Bank";

    const token = localStorage.getItem("budai_token") || "";

    try {
      const response = await fetch("http://localhost:8080/api/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          input: textToSend,
          active_account_id: bankName,
          user_id: localStorage.getItem("budai_token"),
          chat_history: messages,
        }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        aiText += chunk;

        const triggerRegex =
          /\[TRIGGER_([A-Z_]+)_CHART(?:[:]([^\]:]+))?(?:[:]([^\]]+))?\]/g;
        let cleanedReply = aiText;
        let match;

        while ((match = triggerRegex.exec(aiText)) !== null) {
          const rawType = match[1].toLowerCase();
          const targetId = match[2];
          const extraParam = match[3];

          console.log(match);

          let triggeredAction = "";

          if (rawType === "categorized")
            triggeredAction = "categorized_doughnut";
          else if (rawType === "balance") triggeredAction = "balance_forecast";
          else if (rawType === "expense") triggeredAction = "expense_forecast";
          else if (rawType === "cash_flow") triggeredAction = "cash_flow_mixed";
          else if (rawType === "health_radar") triggeredAction = "health_radar";
          else if (rawType.startsWith("historical")) {
            const lowerText = aiText.toLowerCase();
            let timeType = "monthly";
            if (lowerText.includes("daily")) timeType = "daily";
            else if (lowerText.includes("weekly")) timeType = "weekly";
            triggeredAction = `historical_${timeType}`;
          }

          if (triggeredAction) {
            onAiAction(triggeredAction, undefined, targetId, extraParam);
          }
          cleanedReply = cleanedReply.replace(match[0], "");
        }

        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].text = cleanedReply.trim();
          return newMessages;
        });
      }

      if (actionType && !aiText.includes("[TRIGGER_")) {
        onAiAction(actionType);
      }
    } catch (e: unknown) {
      console.log(e);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Critical Engine Error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#161B22] border border-slate-800 rounded-3xl overflow-hidden shadow-2xl">
      <div className="p-4 border-b border-slate-800 bg-[#1c2128] flex items-center shrink-0">
        <span className="font-bold text-xs tracking-widest text-[#00FFAA] flex items-center gap-2">
          <Sparkles size={14} /> BUDAI CHAT
        </span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
            <h2 className="text-xl font-bold text-white">
              Ask BudAI to analyze your finances and generate insights,
              forecasts, and visualizations!
            </h2>
            <div className="grid grid-cols-2 gap-3 w-full max-w-sm">
              <button
                onClick={() =>
                  handleSend("Classify my recent transactions.", "categorized")
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <ShieldCheck
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Categorize Data
              </button>

              <button
                onClick={() =>
                  handleSend(
                    "Generate a 30-day balance forecast.",
                    "balance_forecast",
                  )
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <TrendingUp
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Balance Forecast
              </button>

              <button
                onClick={() =>
                  handleSend(
                    "Generate an expense forecast for the next 30 days.",
                    "expense_forecast",
                  )
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <Activity
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Expense Forecast
              </button>

              <button
                onClick={() =>
                  handleSend("Plot my monthly expenses.", "historical_monthly")
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <CalendarDays
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Monthly History
              </button>

              <button
                onClick={() =>
                  handleSend(
                    "Plot my daily spending trends.",
                    "historical_daily",
                  )
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <BarChart2
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Daily Spending
              </button>

              <button
                onClick={() =>
                  handleSend("What is my highest spending category?", undefined)
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <Flame
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Highest Spend
              </button>
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${m.role === "user" ? "bg-[#00FFAA] text-black font-semibold rounded-br-none shadow-lg" : "bg-[#1c2128] text-slate-200 border border-slate-700/50 rounded-bl-none shadow-md"}`}
              >
                {m.role === "assistant" ? (
                  <div className="prose prose-invert max-w-none prose-sm prose-p:leading-relaxed prose-pre:bg-[#0D1117] prose-li:marker:text-[#00FFAA]">
                    <ReactMarkdown>{m.text}</ReactMarkdown>
                  </div>
                ) : (
                  m.text
                )}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="flex items-center gap-2 text-[#00FFAA] text-xs font-mono px-2">
            <Loader2 className="w-4 h-4 animate-spin" /> BudAI is thinking...
          </div>
        )}
      </div>

      <div className="p-4 bg-[#1c2128] border-t border-slate-800 shrink-0">
        <div className="relative w-full">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Command BudAI..."
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl py-3 pl-4 pr-12 text-sm text-white focus:border-[#00FFAA] outline-none transition-all"
          />
          <button
            onClick={() => handleSend()}
            className="absolute right-2 top-1.5 p-1.5 bg-[#00FFAA] text-black rounded-lg hover:scale-105 transition-transform"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
